#!/bin/bash
# SageMaker BYOC entrypoint for DreamZero LoRA fine-tuning.
#
# Maps the SageMaker filesystem contract onto the torchrun command validated on
# a reference EC2 GPU instance. We deliberately do NOT use the
# sagemaker-training toolkit: channels and hyperparameters are read straight
# from /opt/ml, so behavior is fully deterministic.
#
# Expected input channels (File mode):
#   dataset   -> converted ALOHA-yam LeRobot dataset (data/ meta/ videos/)
#   wan       -> Wan2.1-I2V-14B-480P  (T5/CLIP/VAE .pth + DiT shards)
#   tokenizer -> umt5-xxl tokenizer files (spiece.model + *.json)
#   agibot    -> DreamZero-AgiBot full pretrained checkpoint
set -uo pipefail

DATA=/opt/ml/input/data/dataset
WAN=/opt/ml/input/data/wan
TOK=/opt/ml/input/data/tokenizer
AGIBOT=/opt/ml/input/data/agibot
# Trainer output on the ML data volume: bare /opt/ml/<dir> paths live on the
# small root overlay filesystem, which a multi-GB checkpoint fills (ENOSPC).
# /opt/ml/input/data/* is the big EBS ML volume. This must be a real
# directory, NOT a symlink — the artifact staging below uses
# `find "$OUT" -maxdepth 1 -type f`, which does not follow symlinks, and the
# model.safetensors gate right after it would fail the job.
# S3 mirroring is this script's job (see the sync loop and restore_checkpoint
# below), deliberately not SageMaker's CheckpointConfig: its sync agent can
# fail the whole job with a generic InternalServerError whenever a multi-GB
# save burst lands in its LocalPath — at a mid-run checkpoint save (killing a
# healthy multi-day run at its first save) or during teardown right after
# training succeeded. Measured 4/4 in this account 2026-08-18/19 (eu-central-1,
# ml.g7e.48xlarge; fingerprint = Training -> Failed with no Uploading
# transition and a null ModelArtifacts, while a byte-identical control without
# CheckpointConfig reached Uploading -> Completed).
#
# AWS documents the same symptom and prescribes exactly this workaround:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-troubleshooting-model-parallel.html
#   § "Saving Checkpoints" — "This *could* be caused by a SageMaker AI
#   limitation while uploading the local checkpoint to Amazon S3 during
#   training ... do not use checkpoint_s3_uri", followed by
#   sync_local_checkpoints_to_s3() / sync_s3_checkpoints_to_local() helpers.
# Read that citation honestly: AWS hedges with "could", and the page sits
# under "(Archived) SageMaker model parallelism library v1.x" with its remedy
# snippet gated on smp.local_rank(), whereas we run DeepSpeed ZeRO-2 and have
# never used SMP. So the doc is a matching symptom and a sanctioned
# workaround, NOT a statement about this configuration — the 4/4 above is what
# actually rules CheckpointConfig out here. No published size limit exists to
# appeal to (neither API_CheckpointConfig nor Service Quotas bounds checkpoint
# bytes or object count).
OUT=/opt/ml/input/data/ckpt
HP=/opt/ml/input/config/hyperparameters.json

fail() {
    echo "ENTRYPOINT FAILURE: $1" | tee /opt/ml/output/failure
    exit 1
}

# CRITICAL FIX #3: make sure every pip-installed nvidia lib dir is on
# LD_LIBRARY_PATH (libcudnn_graph.so.9 load error otherwise in some shells).
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
NVLIBS=$(find "$SITE/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"

# CRITICAL FIX #4: EFA-equipped instances (p4de/p5-class — anything where
# SageMaker actually attaches the EFA adapter; g7e reports EfaSupported=true
# but gets none) abort the moment a process calls fork() after Libfabric has
# initialised, because the EFA provider's registered memory is not fork-safe:
#   "A process has executed an operation involving a call to the fork() system
#    call ... Your job will now abort." -> SIGABRT on every rank.
# The trainer's dataloader forks (dataloader_num_workers>0), so this fires
# during the first epoch, after ~5 min of weight loading and a clean NCCL init.
# rdma-core in this image is new enough for fork support, so opting in is all
# that is needed. Harmless on instances without EFA.
export FI_EFA_FORK_SAFE=1

# --- Hyperparameters (SageMaker writes them all as strings) ---
hp() {  # hp <key> <default>
    python - "$1" "$2" <<'EOF'
import json, sys
try:
    with open("/opt/ml/input/config/hyperparameters.json") as f:
        hps = json.load(f)
except FileNotFoundError:
    hps = {}
print(hps.get(sys.argv[1], sys.argv[2]))
EOF
}

MAX_STEPS=$(hp max_steps 1000)
SAVE_STEPS=$(hp save_steps 500)
LR=$(hp learning_rate 1e-5)
BATCH=$(hp per_device_train_batch_size 1)
SEED=$(hp seed 42)
WARMUP=$(hp warmup_ratio 0.05)
FPS_YAM=$(hp fps_yam 50)
RECIPE=$(hp recipe yam)   # yam (GEAR bimanual) | droid (single-arm Franka)
# decord decodes num_frames x num_views frames per sample; a single worker
# starves multi-GPU instances (the loader, not the GPUs, sets the step time)
NUM_WORKERS=$(hp dataloader_num_workers 8)
EXTRA=$(hp extra_overrides "")
# S3 prefix this script mirrors $OUT to (empty disables the sync). The
# launcher passes checkpoints-sync/<job>/ — the same layout CheckpointConfig
# used to produce, so downstream consumers (merge recovery, live loss watch)
# are unchanged. Accepted as the checkpoint_s3_uri hyperparameter (what
# run_pipeline.py sends) or the CKPT_S3_URI environment variable (for
# launchers that set Environment instead) — the hyperparameter wins.
CKPT_S3=$(hp checkpoint_s3_uri "${CKPT_S3_URI:-}")

NUM_GPUS=$(nvidia-smi -L | wc -l)
[ "$NUM_GPUS" -ge 1 ] || fail "no GPUs visible"

# --- Preflight: every path the trainer dereferences must exist ---
for f in \
    "$WAN/models_t5_umt5-xxl-enc-bf16.pth" \
    "$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "$WAN/Wan2.1_VAE.pth" \
    "$TOK/spiece.model" \
    "$AGIBOT/model.safetensors.index.json" \
    "$DATA/meta/modality.json" \
    "$DATA/meta/relative_stats_dreamzero.json"; do
    [ -e "$f" ] || fail "missing required input: $f"
done

# recipe selects the upstream data config + dataset-root override. The yam
# recipe additionally requires the GEAR embodiment.json our prep stage writes;
# DROID datasets (GEAR-Dreams/DreamZero-DROID-Data layout) don't carry one.
case "$RECIPE" in
  yam)
    [ -e "$DATA/meta/embodiment.json" ] || fail "missing required input: $DATA/meta/embodiment.json"
    grep -q '"embodiment_tag": *"yam"' "$DATA/meta/embodiment.json" || fail "dataset embodiment_tag is not yam"
    DATA_CFG="dreamzero/yam_relative"
    DATA_ROOT_ARGS=(yam_data_root="$DATA" fps.yam="$FPS_YAM")
    ;;
  droid)
    DATA_CFG="dreamzero/droid_relative"
    DATA_ROOT_ARGS=(droid_data_root="$DATA")
    ;;
  *) fail "unknown recipe '$RECIPE' (expected yam or droid)" ;;
esac

mkdir -p "$OUT"
cd /opt/ml/code/dreamzero

# --- Spot-restart resume: the DOWNLOAD half of the checkpoint contract ---
# On a managed-spot interruption SageMaker relaunches this exact job spec, so
# $CKPT_S3 (derived from the job name) already holds the interrupted attempt's
# saves — but nothing stages them back into $OUT, so without this the relaunch
# trains from step 0. That is not merely wasted spend: the restarted attempt
# re-uploads checkpoint-N/ over the SAME keys in a non-versioned bucket, so the
# older, better-trained recovery point is destroyed. Measured 2026-08-30 on a
# 15,000-step managed-spot run: ~3,900 steps of attempt 1 (20.83 h, ~$480)
# overwritten by attempt 2's from-scratch weights.
#
# Upstream needs no flag for this: base.py calls get_checkpoint_path($OUT),
# which globs checkpoint-* and resumes from the highest N (utils.py:53-73).
# Three properties of that function drive every guard below:
#   1. a top-level config.json makes it return continue_training=False, so the
#      trainer prints "Models is ready ... Skip training", exits 0, and the job
#      reports SUCCESS having trained nothing. Never restore top-level files
#      other than the loss log.
#   2. it picks the highest-numbered dir with NO integrity check, so a torn
#      checkpoint would be selected in preference to a good older one. We
#      restore exactly ONE pre-verified directory, which makes "highest" and
#      "verified" the same dir by construction.
#   3. no checkpoint-* at all -> (None, True) -> normal first-time training, so
#      a fresh run passes through here untouched.
# Pull the mirror's loss_log.jsonl down before training starts. The trainer
# APPENDS to this file, and phase 1 of the up-sync loop mirrors it every 60s
# without --delete-style reconciliation, so whatever is local wins in S3. Without
# this the first tick replaces a long history with a few lines. Needed on both
# paths that continue into training with an existing mirror: a real resume, and
# the all-candidates-rejected restart.
fetch_loss_log() {  # fetch_loss_log <bucket> <key_prefix>
    aws s3 cp "s3://$1/${2}loss_log.jsonl" "$OUT/loss_log.jsonl" \
        --only-show-errors 2>/dev/null \
        && echo "restore: loss_log.jsonl restored (expect duplicate step numbers across the resume)"
    return 0
}

restore_checkpoint() {
    # FIRST, before any early return. A top-level config.json in $OUT makes
    # get_checkpoint_path report continue_training=False, so the trainer prints
    # "Models is ready ... Skip training" and exit(0)s -- and then the final copy
    # below lifts whatever model.safetensors is sitting there into /opt/ml/model,
    # the gate finds it, and the job reports SUCCESS having trained nothing.
    # Nothing this function downloads can create that pair (the restore path
    # deliberately fetches neither), so at start-up they can only have arrived on
    # a mounted `ckpt` channel staged from a previous run's OUTPUT prefix, where
    # the final save writes them together. That used to be checked after both
    # early returns, i.e. on the one path that can never produce them.
    for marker in config.json model.safetensors; do
        if [ -e "$OUT/$marker" ]; then
            rm -f "$OUT/$marker"
            echo "restore: removed a top-level $marker from $OUT — it came from a" \
                 "mounted channel, and leaving it would make the trainer skip" \
                 "training and report SUCCESS with the staged weights"
        fi
    done

    [ -n "$CKPT_S3" ] || return 0

    # A mounted `ckpt` input channel (manual chained resume) lands in $OUT too.
    # It is the operator's explicit choice, so it wins over anything in S3.
    # Known consequence, deliberately kept: on a managed-spot relaunch SageMaker
    # re-downloads that channel from the S3Uri fixed at submit time, so a
    # long-running job that has since mirrored newer checkpoints will still see
    # the ORIGINAL mount here and resume from it. If you chain a resume onto a
    # spot job, expect the relaunch to restart from the staged step, not from the
    # newest step reached.
    if compgen -G "$OUT/checkpoint-*" > /dev/null 2>&1; then
        echo "restore: $OUT already contains a checkpoint (mounted ckpt channel) — using it as-is"
        return 0
    fi

    local bucket key_prefix listing
    bucket=$(printf '%s' "$CKPT_S3" | sed -E 's#^s3://([^/]+)(/.*)?$#\1#')
    # Normalise to either "" (bucket root) or "a/b/". The earlier one-liner turned
    # a bucket-root URI -- s3://bkt or s3://bkt/, both documented above as valid
    # CKPT_S3_URI values -- into "/", and no S3 key begins with a slash, so the
    # listing matched nothing and every relaunch silently restarted from step 0
    # while the uploader was writing checkpoint-*/ at the root. An empty prefix is
    # correct for that case: list-objects-v2 and the selector both handle it.
    key_prefix=$(printf '%s' "$CKPT_S3" | sed -E 's#^s3://[^/]+##; s#^/+##; s#/+$##')
    if [ -n "$key_prefix" ]; then key_prefix="$key_prefix/"; fi
    [ -n "$bucket" ] && [ "$bucket" != "$CKPT_S3" ] || {
        echo "restore: cannot parse a bucket out of '$CKPT_S3' — skipping restore"
        return 0
    }

    listing=/tmp/ckpt_listing.json
    # Default CLI behaviour paginates in full; 500-step saves make this a few
    # hundred keys.
    #
    # This must NOT fail open. A first-ever attempt does not reach the error
    # branch: an empty/nonexistent prefix LISTS SUCCESSFULLY with KeyCount 0, so
    # a listing error is always a real failure (throttling, transient S3, a
    # broken role) — and those are exactly the cases where assuming "no
    # checkpoint exists" would train from step 0 and overwrite a recovery point
    # that may well be sitting there. Same job name means the same prefix, and
    # the bucket is not versioned, so that overwrite is unrecoverable. Retry,
    # then refuse to run: a failed job costs minutes, a silent from-scratch
    # relaunch cost ~$480 and ~3,900 steps on 2026-08-30.
    listed=0
    for attempt in 1 2 3 4 5; do
        if aws s3api list-objects-v2 --bucket "$bucket" --prefix "$key_prefix" \
                --output json > "$listing" 2>/tmp/ckpt_listing.err; then
            listed=1
            break
        fi
        echo "restore: listing attempt $attempt failed: $(tail -1 /tmp/ckpt_listing.err 2>/dev/null)"
        [ "$attempt" -lt 5 ] && sleep 15
    done
    if [ "$listed" -ne 1 ]; then
        fail "restore: cannot list s3://$bucket/$key_prefix after 5 attempts. Refusing to
         start, because a checkpoint may exist there and training from step 0 would
         overwrite it irrecoverably (same job name -> same prefix, bucket versioning off).
         A first-ever attempt does NOT reach this path — an empty prefix lists fine with
         KeyCount 0 — so this is a real S3/permissions failure, not a fresh run."
    fi

    # Completeness is judged against THIS instance's GPU count on purpose:
    # zero2.json sets load_universal=false, so DeepSpeed can only resume into
    # the same world size it was saved from. A shard-count mismatch here is a
    # checkpoint we must not hand to the trainer, not a checkpoint to repair.
    local step manifest=/tmp/ckpt_manifest.txt candidates=/tmp/ckpt_candidates.txt
    rm -f "$manifest" "$candidates"
    step=$(python - "$listing" "$key_prefix" "$NUM_GPUS" "$manifest" "$candidates" <<'PY'
import json, re, sys

path, prefix, ngpu, manifest = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
candidates = sys.argv[5]
with open(path) as fh:
    body = json.load(fh)

ck = {}
for obj in body.get("Contents") or []:
    key = obj["Key"]
    rel = key[len(prefix):] if key.startswith(prefix) else key
    m = re.match(r"^checkpoint-(\d+)/(.+)$", rel)
    if m:
        ck.setdefault(int(m.group(1)), {})[m.group(2)] = obj["Size"]

# Record what was on offer BEFORE any of it is judged. An empty stdout means
# "nothing to resume from" either way, but the two ways differ: an empty prefix
# is a first-ever attempt, whereas candidates-but-all-rejected means this job
# name has run before and its checkpoints are unusable to this trainer. The
# shell cannot tell those apart from the exit code, and it must not fail closed
# on the second one -- see the comment at the `-z "$step"` branch.
if ck:
    with open(candidates, "w") as fh:
        for n in sorted(ck, reverse=True):
            fh.write(f"{n}\n")


def norm(name):
    """Fold the step number out so the same logical file compares across checkpoints."""
    return re.sub(r"^global_step\d+/", "global_step/", name)


# A TRUNCATED object is the failure a presence-and-zero-byte check cannot see. The
# 60s up-sync loop can stat mp_rank_00_model_states.pt while DeepSpeed rank 0 is
# still writing it; s3transfer takes the transfer size once at submit and never
# re-stats, so that tick lands a well-formed, non-zero, SHORT object. Normally a
# later tick fixes it, but a spot reclaim inside that window leaves S3 holding a
# checkpoint with every name present, nothing zero-byte, 8 shards -- and a short
# module state. Resuming from it dies in torch.load ~10 min in, and because
# `Failed` is not relaunched by managed spot, the run is simply dead while the
# intact checkpoint 500 steps below it was never tried.
#
# No hardcoded sizes needed: within one run these files are byte-invariant
# (verified across all 39 checkpoints in this bucket -- 24 of 26 files identical
# per run). So the largest size seen for a logical file IS its true size.
# Deliberately NOT size-checked: `latest` (13/14/15 B as the step count gains
# digits) and trainer_state.json (grows with log_history) legitimately vary, and
# comparing them would false-reject a good checkpoint.
# Limitation, stated rather than hidden: with only ONE checkpoint in the prefix
# there is nothing to compare against, so the zero-byte test is the only guard
# there. That is why it is kept as well, not replaced.
SIZE_CHECKED = re.compile(r"^(global_step/.+|scheduler\.pt|rng_state_\d+\.pth)$")
expected = {}
for files in ck.values():
    for name, size in files.items():
        n = norm(name)
        if SIZE_CHECKED.match(n) and size > expected.get(n, -1):
            expected[n] = size

for n in sorted(ck, reverse=True):
    files, missing = ck[n], []
    # scheduler.pt is REQUIRED even though it looks incidental: on the DeepSpeed
    # path transformers 4.51.3 torch.loads it with no os.path.isfile guard (the
    # guard exists only in the non-DeepSpeed branch), and zero2.json declares no
    # scheduler, so self.lr_scheduler is a plain LambdaLR and that branch is
    # taken. A missing scheduler.pt is therefore FileNotFoundError on all 8 ranks
    # ~10 min into the resume, and every manual retry fails identically. Falling
    # through to checkpoint-(N-500) costs 500 steps instead of the run.
    required = ["latest", "trainer_state.json", "scheduler.pt",
                f"global_step{n}/mp_rank_00_model_states.pt"]
    required += [f"global_step{n}/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt"
                 for r in range(ngpu)]
    for req in required:
        if req not in files:
            missing.append(f"no {req}")
    empty = [k for k, size in files.items() if size == 0]
    if empty:
        missing.append(f"{len(empty)} zero-byte object(s) e.g. {sorted(empty)[0]}")
    for name, size in sorted(files.items()):
        want = expected.get(norm(name))
        if want is not None and size < want:
            missing.append(f"{name} is {size} B, expected {want} B (truncated upload)")
    if missing:
        print(f"restore:   reject checkpoint-{n}: {'; '.join(missing[:4])}", file=sys.stderr)
        continue
    # Hand the shell the exact sizes so the DOWNLOAD can be verified too, not
    # just the S3 side.
    with open(manifest, "w") as fh:
        for name, size in sorted(files.items()):
            fh.write(f"{size}\t{name}\n")
    print(n)
    break
PY
)
    selrc=$?

    # Same rule as the listing above: only a CLEAN "found nothing" may fall
    # through to step-0 training. A crashed selector (malformed JSON, a python
    # that is not on PATH, an unexpected key) also yields an empty $step, and
    # treating that as "no checkpoint exists" is how a recovery point gets
    # overwritten by a fresh run.
    if [ "$selrc" -ne 0 ]; then
        fail "restore: the checkpoint selector exited $selrc instead of reporting a result.
         Refusing to start: an empty answer from a crashed selector is indistinguishable
         from 'no checkpoint', and training from step 0 would overwrite whatever is in
         s3://$bucket/$key_prefix irrecoverably."
    fi

    if [ -z "$step" ]; then
        if [ -s "$candidates" ]; then
            # Candidates existed and every one was rejected (reasons on stderr
            # just above). This is deliberately NOT a `fail`, unlike every other
            # ambiguity in this function: a rejected checkpoint is by
            # construction unresumable by this trainer, managed spot does not
            # relaunch a Failed job, and a relaunch under the same name would
            # re-read the same prefix and fail identically -- so failing closed
            # would make the run permanently unstartable in exchange for
            # protecting bytes that cannot be resumed. What it IS is worth
            # shouting about, because from here on the up-sync loop will
            # overwrite those keys as this attempt re-passes each step, and the
            # bucket is not versioned. If the rejections are shard-count
            # mismatches, salvage them with zero_to_fp32.py BEFORE that happens.
            echo "restore: WARNING — $(wc -l < "$candidates") checkpoint(s) exist under" \
                 "s3://$bucket/$key_prefix but NONE is complete for this instance" \
                 "($NUM_GPUS GPU(s)): $(tr '\n' ' ' < "$candidates")"
            echo "restore: WARNING — training from step 0. These keys will be overwritten" \
                 "as this attempt reaches each step, and the bucket has no versioning." \
                 "Copy anything you want to salvage out now."
        else
            echo "restore: no checkpoint under s3://$bucket/$key_prefix — training from scratch"
        fi
        # Both branches. Without this, phase 1 of the up-sync replaces the mirror's
        # long loss_log.jsonl with this attempt's short one and the earlier
        # attempt's metrics are gone -- `aws s3 sync` re-uploads on ANY size
        # mismatch, including smaller, and the bucket is not versioned. It is
        # needed on the plain "no checkpoint" path too, because that does NOT
        # imply an empty prefix: phase 1 mirrors loss_log.jsonl from the first 60s
        # tick while phase 2 needs `latest` plus 2 minutes of quiescence, so an
        # attempt reclaimed before its first save (save_steps=500 is >4 h at
        # ~30 s/step) leaves hundreds of lines of history and zero checkpoint-*
        # keys. A miss here costs one failed HeadObject.
        fetch_loss_log "$bucket" "$key_prefix"
        return 0
    fi

    # Only the one verified directory, plus the loss log so the run's history
    # survives (the trainer appends to it, and the up-sync loop would otherwise
    # replace the longer S3 copy with this attempt's short one). Restoring
    # top-level config.json / model.safetensors is what triggers "Skip
    # training", so they are never fetched.
    echo "restore: newest COMPLETE checkpoint is checkpoint-$step — downloading"
    local t0=$SECONDS ok=0
    for attempt in 1 2 3; do
        if aws s3 sync "s3://$bucket/${key_prefix}checkpoint-$step/" \
                       "$OUT/checkpoint-$step/" --only-show-errors; then
            ok=1
            break
        fi
        echo "restore: download attempt $attempt failed"
        [ "$attempt" -lt 3 ] && sleep 15
    done
    if [ "$ok" -ne 1 ]; then
        rm -rf "$OUT/checkpoint-$step"
        fail "restore: could not download checkpoint-$step after 3 attempts. Refusing to
         continue, because training from step 0 would overwrite this checkpoint in
         $CKPT_S3 with from-scratch weights and destroy the only recovery point."
    fi

    # Re-verify on local disk BYTE-EXACTLY against the manifest the selector
    # emitted. `[ -s ]` (non-empty) is not enough for the same reason the S3 gate
    # needed sizes: a truncated 85 GiB file is non-empty and looks fine, and a
    # download can be cut short by anything from a spot reclaim to a full disk.
    local bad=0 n_checked=0 size want name
    if [ ! -s "$manifest" ]; then
        echo "restore: selector produced no manifest — cannot verify the download"
        bad=1
    else
        while IFS=$'\t' read -r want name; do
            [ -n "${name:-}" ] || continue
            size=$(stat -c %s "$OUT/checkpoint-$step/$name" 2>/dev/null || echo missing)
            if [ "$size" != "$want" ]; then
                echo "restore: $name is $size B on disk, S3 says $want B"
                bad=1
            fi
            n_checked=$((n_checked + 1))
        done < "$manifest"
        echo "restore: verified $n_checked file(s) byte-for-byte against the S3 listing"
    fi
    # `latest` names the global_step dir DeepSpeed will open. A right-length file
    # with wrong content would pass the size check and then make
    # deepspeed_load_checkpoint raise AFTER the 86 GiB download and ~10 min of
    # loading, so assert the content, not just the bytes.
    if ! grep -qx "global_step$step" "$OUT/checkpoint-$step/latest" 2>/dev/null; then
        echo "restore: latest does not say 'global_step$step' (got: $(head -c 64 "$OUT/checkpoint-$step/latest" 2>/dev/null | tr -d '\n'))"
        bad=1
    fi
    # Deliberately NOT required: the 8 rng_state_*.pth. _load_rng_state does guard
    # with isfile and only logs "Didn't find an RNG file", so absence costs
    # bit-exactness rather than the run — and upstream throws that away anyway,
    # since base.py forces ignore_data_skip=True and reseeds to seed+global_step,
    # printing that the resume is non-reproducible. Requiring them would add
    # false-reject risk for a property we do not get either way.
    if [ "$bad" -ne 0 ]; then
        rm -rf "$OUT/checkpoint-$step"
        fail "restore: checkpoint-$step verified in S3 but arrived incomplete; refusing to
         train from step 0 over a good recovery point."
    fi
    # The skip-training markers are cleared at the top of this function instead of
    # here, so that the mounted-channel path gets the same treatment. Nothing this
    # function downloads can recreate them: the sync above targets
    # $OUT/checkpoint-$step/ only.

    fetch_loss_log "$bucket" "$key_prefix"

    # The 60s up-sync loop below does NOT re-upload any of this. Verified
    # empirically: `aws s3 sync` in the download direction sets each local
    # mtime to the S3 object's LastModified (not to "now"), so the reverse sync
    # sees equal size and a not-newer source and skips all 26 objects —
    # `aws s3 sync "$OUT" "$CKPT_S3" --dryrun` right after a restore wants to
    # upload 0. Do not "optimise" this with --size-only or an --exclude; it is
    # already a no-op, and --size-only would weaken the normal save path.
    # Make a silent step-0 restart visible in CloudWatch instead of only in the
    # loss curve: print the step the restored state actually claims, so it can be
    # compared against the trainer's own "Resuming training from ..." line.
    echo "restore: trainer_state.json reports global_step=$(python - "$OUT/checkpoint-$step/trainer_state.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("global_step", "?"))
except Exception as e:
    print(f"unreadable ({type(e).__name__})")
PY
)"
    echo "restore: checkpoint-$step in place in $((SECONDS - t0))s — the trainer will resume from it"
}

restore_checkpoint

# Background checkpoint mirror: every 60s, push whatever the trainer has
# written (loss_log.jsonl streams live; checkpoints survive a mid-run crash).
# The aws CLI resolves credentials from the SageMaker execution role fresh on
# every invocation, so multi-day runs never hit token expiry, and a transient
# S3 error just waits for the next tick. Unlike CheckpointConfig's agent
# there is nothing here a mid-run save burst can kill, and the teardown case
# is closed by final_sync below, which completes BEFORE the container exits.
# setsid: the loop gets its own process group (pgid == pid), so final_sync
# can kill the loop AND any in-flight aws child with one group signal. Killing
# only the subshell would orphan a mid-transfer aws process, which may have
# stat'ed a checkpoint file mid-write and would then land a truncated object
# in S3 AFTER the verified final sync — silently corrupting the recovery path.
SYNC_PID=""
if [ -n "$CKPT_S3" ]; then
    # Two-phase on purpose. A single `aws s3 sync "$OUT" "$CKPT_S3"` can stat
    # global_step<N>/mp_rank_00_model_states.pt while DeepSpeed rank 0 is still
    # writing it — the save takes ~4 min for 85 GiB — and s3transfer takes the
    # transfer size ONCE at submit and never re-stats, so that tick lands a
    # well-formed, non-zero, SHORT object. A later tick normally corrects it, but
    # a spot reclaim inside that window leaves S3 holding a checkpoint whose every
    # name is present, nothing is zero-byte, all shards are there, and the module
    # state is truncated. That was harmless while nothing ever read the mirror
    # back; restore_checkpoint reads it now, so the producer needs the integrity
    # contract rather than only the consumer.
    #   phase 1: everything outside checkpoint-*/ (loss_log.jsonl must stream live)
    #   phase 2: only checkpoint dirs that have SETTLED — `latest` written (the
    #            trainer writes it last) and nothing touched in the last 2 minutes
    setsid bash -c '
    OUT=$1; DST=${2%/}/
    while true; do
        sleep 60
        aws s3 sync "$OUT" "$DST" --exclude "checkpoint-*/*" --only-show-errors || true
        for d in "$OUT"/checkpoint-*/; do
            [ -d "$d" ] || continue
            [ -f "${d}latest" ] || continue
            [ -z "$(find "$d" -type f -mmin -2 -print -quit 2>/dev/null)" ] || continue
            rel=${d#"$OUT"/}
            aws s3 sync "$d" "$DST$rel" --only-show-errors || true
        done
    done' _ "$OUT" "$CKPT_S3" &
    SYNC_PID=$!
    echo "checkpoint sync loop started (60s): $OUT -> $CKPT_S3"
fi

final_sync() {
    [ -n "$CKPT_S3" ] || return 0
    if [ -n "$SYNC_PID" ]; then
        # SIGTERM, and do NOT "improve" this to SIGINT. It looks like SIGINT
        # would be better -- awscli converts only SIGINT into an s3transfer
        # cancellation, and cancellation is what runs the registered
        # AbortMultipartUpload, so SIGTERM leaves up to ~85 GiB of uploaded parts
        # in the bucket, billed and invisible to `aws s3 ls` and ListObjectsV2
        # alike. It does not work, and it HANGS THE JOB: bash sets SIGINT to
        # IGNORED in a command started asynchronously with `&` while job control
        # is off, and children inherit that disposition, so the loop ignores it
        # and so does the `aws` child. Measured 2026-09-01 on
        # a 20-step smoke job: the loop survived SIGINT, the `wait`
        # never returned, and the job sat in Training for 87 minutes after
        # torchrun exited -- no final_sync, no artifacts -- until it was stopped
        # by hand. Reproduced in isolation: survived SIGINT, died on SIGTERM.
        # The orphaned-parts problem is real but cannot be solved from here (the
        # SIGKILL below is uncatchable too, and a spot reclaim never reaches
        # final_sync at all), which is why the bucket carries an
        # AbortIncompleteMultipartUpload lifecycle rule instead. That is the
        # actual remedy; see AssetsBucket in cdk/dreamzero_pipeline/pipeline_stack.py.
        kill -TERM -- "-$SYNC_PID" 2>/dev/null   # whole process group
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 -- "-$SYNC_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SYNC_PID" 2>/dev/null
        # AFTER the escalation, never before it. `wait` on a process group that
        # ignored the signal blocks forever, and nothing can ignore the SIGKILL
        # above -- gating it behind an unbounded `wait` is what turned a bad
        # signal choice into a hung job.
        wait "$SYNC_PID" 2>/dev/null
    fi
    for attempt in 1 2 3 4 5; do
        if aws s3 sync "$OUT" "$CKPT_S3" --only-show-errors; then
            echo "final checkpoint sync verified: $CKPT_S3"
            return 0
        fi
        echo "final checkpoint sync attempt $attempt failed"
        [ "$attempt" -lt 5 ] && { echo "  retrying in 15s"; sleep 15; }
    done
    # Do not fail the job over this: on a successful run the LoRA also ships
    # in model.tar.gz, which is the merge stage's primary source.
    echo "WARNING: final checkpoint sync to $CKPT_S3 failed after 5 attempts —"
    echo "         S3 may be missing the last save."
    # Said conditionally, because final_sync is called before the RC gate and this
    # message is therefore reachable on a crashed run too -- where the fallback it
    # used to promise unconditionally does not exist. `fail` exits before the
    # /opt/ml/model copy, and SageMaker uploads no model artifact for a failed job,
    # so on the crash path there is no model.tar.gz to fall back to and S3 is the
    # only copy there ever was.
    if [ "${RC:-1}" -eq 0 ]; then
        echo "         model.tar.gz still carries the LoRA — use that instead"
    else
        echo "         and because this run FAILED there is no model.tar.gz either:"
        echo "         no model artifact is uploaded for a failed job. Whatever the"
        echo "         60s loop already mirrored is the only copy — check it before"
        echo "         relaunching, because a relaunch reuses this prefix."
    fi
    return 1
}

echo "=== DreamZero SageMaker training ==="
echo "GPUs=$NUM_GPUS  recipe=$RECIPE  max_steps=$MAX_STEPS  save_steps=$SAVE_STEPS  lr=$LR  batch=$BATCH"
nvidia-smi | head -15

# Gotchas encoded below: save_total_limit>=5 is asserted by the
# repo; wandb_project is mandatory even with report_to=none. fps.yam records
# the dataset rate but is inert at train time under the decord video backend
# (which every dreamzero data recipe pins): frames are selected by matching
# the parquet timestamp column to video PTS, and action chunking is pure
# frame-index arithmetic. The load-bearing invariant is that alignment —
# video frame i must correspond to parquet row i — which the prep/GEAR
# conversion establishes. fps.yam would only take effect under
# video_backend=torchcodec, so it is still passed through faithfully.
torchrun --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
    report_to=none wandb_project=dreamzero \
    data="$DATA_CFG" \
    train_architecture=lora \
    num_frames=33 action_horizon=24 num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 num_action_per_block=24 num_state_per_block=1 \
    seed="$SEED" \
    training_args.learning_rate="$LR" \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    training_args.warmup_ratio="$WARMUP" \
    output_dir="$OUT" \
    per_device_train_batch_size="$BATCH" \
    max_steps="$MAX_STEPS" save_steps="$SAVE_STEPS" save_total_limit=5 save_strategy=steps \
    weight_decay=1e-5 upload_checkpoints=false \
    bf16=true tf32=true eval_bf16=true \
    dataloader_pin_memory=false dataloader_num_workers="$NUM_WORKERS" \
    image_resolution_width=320 image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 frame_seqlen=880 \
    "${DATA_ROOT_ARGS[@]}" \
    dit_version="$WAN" \
    text_encoder_pretrained_path="$WAN/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN/Wan2.1_VAE.pth" \
    tokenizer_path="$TOK" \
    pretrained_model_path="$AGIBOT" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    $EXTRA
RC=$?

# Sync in BOTH outcomes, before the exit-code gate: after a crash, the partial
# checkpoints in S3 are exactly what a post-mortem (or a merge of an earlier
# complete save) needs, and nothing may exist only on this disposable instance.
final_sync

if [ $RC -ne 0 ]; then
    fail "torchrun exited with code $RC"
fi

# Final artifacts (LoRA safetensors + configs, ~220MB) -> model.tar.gz.
# checkpoint-*/ stays out: those are DeepSpeed resume states, already synced
# to S3 by the loop above.
echo "Training done; copying final artifacts to /opt/ml/model"
mkdir -p /opt/ml/model
find "$OUT" -maxdepth 1 -type f -exec cp {} /opt/ml/model/ \;
[ -d "$OUT/experiment_cfg" ] && cp -r "$OUT/experiment_cfg" /opt/ml/model/
[ -f /opt/ml/model/model.safetensors ] || fail "training finished but model.safetensors missing in output_dir"
echo "=== done ==="
