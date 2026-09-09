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
# Trainer output goes to the CheckpointConfig LocalPath. The launcher sets
# CheckpointConfig{LocalPath=/opt/ml/checkpoints, S3Uri=checkpoints-sync/<job>/}
# on every training job, so SageMaker mounts this path on the data volume,
# mirrors whatever the trainer writes there to S3 while the job runs, and on a
# managed-spot relaunch restores the prefix into it before this script starts.
# Checkpoints are a few GB each (docker/patches/0004 keeps the frozen base out
# of them), so the mirror is cheap and a save lands in S3 within seconds.
# This must be a real directory, NOT a symlink — the artifact staging below
# uses `find "$OUT" -maxdepth 1 -type f`, which does not follow symlinks.
OUT=/opt/ml/checkpoints
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

# --- Checkpoints: SageMaker's CheckpointConfig owns the S3 mirror ---
# On a managed-spot relaunch SageMaker re-runs this exact job spec and restores
# the CheckpointConfig prefix into $OUT before the container starts. Upstream
# then resumes on its own: base.py calls get_checkpoint_path($OUT), which picks
# the highest-numbered checkpoint-*/ (utils.py:53-73). Two properties of that
# function need a guard in front of it:
#   1. it does NO integrity check. The sync agent mirrors files as they land,
#      so a reclaim during a save leaves S3 — and therefore $OUT — holding a
#      torn checkpoint-N/, which would be chosen over the good one below it and
#      die in torch.load ten minutes into the relaunch. Managed spot does not
#      relaunch a Failed job, so that kills the run.
#   2. a top-level config.json makes it report "Models is ready ... Skip
#      training" and exit 0, and the job then reports SUCCESS having trained
#      nothing. A finished run's prefix (pointed at deliberately to continue a
#      run) restores exactly that pair.
# Completeness is judged against THIS instance's GPU count: zero2.json sets
# load_universal=false, so DeepSpeed can only resume into the world size it
# was saved from. Sizes are cross-checked between checkpoints because the
# per-file sizes are byte-invariant within a run, which catches a truncated
# object that is present and non-empty. rng_state_*.pth are not required:
# transformers only logs their absence, and upstream reseeds on resume anyway.
prune_checkpoints() {
    if ! compgen -G "$OUT/checkpoint-*" > /dev/null 2>&1 && [ ! -e "$OUT/config.json" ]; then
        echo "checkpoints: none restored into $OUT — training from scratch"
        return 0
    fi
    python - "$OUT" "$NUM_GPUS" "$MAX_STEPS" <<'PY' || fail "checkpoint guard crashed — refusing to guess which restored checkpoint is safe to resume from"
import json, os, re, shutil, sys

out, ngpu, max_steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ck = {}
for d in os.listdir(out):
    m = re.fullmatch(r"checkpoint-(\d+)", d)
    if not m or not os.path.isdir(os.path.join(out, d)):
        continue
    base, files = os.path.join(out, d), {}
    for root, _, names in os.walk(base):
        for n in names:
            path = os.path.join(root, n)
            files[os.path.relpath(path, base)] = os.path.getsize(path)
    ck[int(m.group(1))] = files


def norm(name):
    """Fold the step number out so the same logical file compares across checkpoints."""
    return re.sub(r"^global_step\d+/", "global_step/", name)


SIZE_CHECKED = re.compile(r"^(global_step/.+|scheduler\.pt|rng_state_\d+\.pth|model\.safetensors)$")
expected = {}
for files in ck.values():
    for name, size in files.items():
        n = norm(name)
        if SIZE_CHECKED.match(n) and size > expected.get(n, -1):
            expected[n] = size

kept = []
for n in sorted(ck, reverse=True):
    files, why = ck[n], []
    # scheduler.pt is required: on the DeepSpeed path transformers 4.51.3
    # torch.loads it with no isfile guard, so its absence is a FileNotFoundError
    # on every rank ~10 min into the resume.
    required = ["latest", "trainer_state.json", "scheduler.pt", "model.safetensors",
                "config.json", f"global_step{n}/mp_rank_00_model_states.pt"]
    required += [f"global_step{n}/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt"
                 for r in range(ngpu)]
    why += [f"no {r}" for r in required if r not in files]
    why += [f"{k} is empty" for k, s in sorted(files.items()) if s == 0]
    why += [f"{k} is {s} B, expected {expected[norm(k)]} B (truncated)"
            for k, s in sorted(files.items()) if norm(k) in expected and s < expected[norm(k)]]
    try:
        with open(os.path.join(out, f"checkpoint-{n}", "latest")) as fh:
            if fh.read().strip() != f"global_step{n}":
                why.append(f"latest does not name global_step{n}")
    except OSError:
        pass
    if why:
        print(f"checkpoints: removing checkpoint-{n}: " + "; ".join(why[:4]), file=sys.stderr)
        shutil.rmtree(os.path.join(out, f"checkpoint-{n}"))
    else:
        kept.append(n)
# Top-level config.json + model.safetensors are what a FINISHED run leaves in
# its output dir, and upstream's get_checkpoint_path reads a top-level
# config.json as "model is ready, skip training" (exit 0). Two cases share
# that shape and must be told apart by the newest checkpoint's global_step:
#   - a relaunch of a run that had already reached max_steps (a reclaim in the
#     seconds between the last save and teardown): keep the markers, let the
#     trainer skip, and the artifact copy below stages the finished weights —
#     retraining would burn a full model load to run one extra step;
#   - a deliberate continuation (the same prefix with a HIGHER max_steps) or
#     markers with no complete checkpoint behind them: remove the markers, or
#     the job reports SUCCESS having trained nothing.
markers = [m for m in ("config.json", "model.safetensors") if os.path.exists(os.path.join(out, m))]
done_step = None
if kept:
    try:
        with open(os.path.join(out, f"checkpoint-{max(kept)}", "trainer_state.json")) as fh:
            done_step = json.load(fh).get("global_step")
    except (OSError, ValueError):
        done_step = None
if len(markers) == 2 and done_step is not None and done_step >= max_steps:
    print(f"checkpoints: training already complete — checkpoint-{max(kept)} reports "
          f"global_step={done_step} >= max_steps={max_steps}; leaving the finished "
          "model.safetensors + config.json in place, the trainer will skip training "
          "and they are staged as the job's artifacts")
else:
    for m in markers:
        os.remove(os.path.join(out, m))
        print(f"checkpoints: removed top-level {m} — it belongs to a finished run "
              f"(newest complete checkpoint reports global_step={done_step}, max_steps={max_steps}); "
              "leaving it would make the trainer skip training and report SUCCESS")
    if kept:
        print("checkpoints: resumable = " + ", ".join(f"checkpoint-{n}" for n in sorted(kept))
              + f" — the trainer will resume from checkpoint-{max(kept)}")
    else:
        print("checkpoints: no complete checkpoint restored — training from scratch")
PY
}

prune_checkpoints

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

if [ $RC -ne 0 ]; then
    fail "torchrun exited with code $RC"
fi

# Final artifacts (LoRA safetensors + configs, ~220MB) -> model.tar.gz.
# checkpoint-*/ stays out: those are DeepSpeed resume states, mirrored to S3
# by CheckpointConfig as they were written.
echo "Training done; copying final artifacts to /opt/ml/model"
mkdir -p /opt/ml/model
find "$OUT" -maxdepth 1 -type f -exec cp {} /opt/ml/model/ \;
[ -d "$OUT/experiment_cfg" ] && cp -r "$OUT/experiment_cfg" /opt/ml/model/
[ -f /opt/ml/model/model.safetensors ] || fail "training finished but model.safetensors missing in output_dir"
echo "=== done ==="
