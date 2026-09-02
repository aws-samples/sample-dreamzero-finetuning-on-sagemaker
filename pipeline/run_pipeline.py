#!/usr/bin/env python3
"""DreamZero fine-tuning pipeline: LeRobot dataset -> servable weights in S3.

Stages (this script is the glue):
  1. fetch    — download the dataset from HuggingFace (skipped for local paths)
  2. detect   — read meta/info.json codebase_version
  3. convert  — v3 -> v2.1 if needed (external converter, dataset-specific)
  4. validate — dataset vs embodiment config: slice bounds, cameras, fps,
                task annotations, action-vs-state degeneracy, and episode
                completeness (validate_dataset.py)
  5. prep     — v2.1 -> GEAR/yam via prep_dataset.py + embodiment config
  6. stage    — dataset -> s3://<bucket>/sagemaker/datasets/<name>/
  7. smoke    — SageMaker job, compute.smoke.steps (gate: a cheap short run
                has to succeed before the expensive one is submitted)
  8. train    — SageMaker job, training.max_steps (or --max-steps)
  9. merge    — LoRA -> AgiBot merge job; servable weights ->
                models/<name>-merged/

Per-run settings (dataset source, compute, hyperparameters) come from
project_config.json at the repo root; CLI flags override it. With the
shipped defaults this is enough:

  python3 run_pipeline.py --name aloha-demo

or point at your own data explicitly:

  python3 run_pipeline.py --hf-dataset lerobot/aloha_static_screw_driver \\
      --name aloha-demo --config configs/aloha_bimanual_14dim.yaml
  python3 run_pipeline.py --dataset /path/to/lerobot_ds --name my-robot-v1 \\
      --config configs/my_robot.yaml [--skip-smoke] [--dry-run]

Resumable: pass --start-at <stage> to skip completed stages.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from pipeline_config import (boto_session, job_tags, load_config,
                             load_project_config)
from solution import get_client, user_agent_string

HERE = Path(__file__).resolve().parent
STAGES = ["fetch", "detect", "convert", "validate", "prep", "stage",
          "smoke", "train", "merge"]

# resolved from the CDK stack outputs (pipeline_config.json) or DREAMZERO_* env.
# --help must work on a fresh clone, before any of those exist — skip the
# config load (and its SSM fallback's network call) when only help is asked.
if any(a in ("-h", "--help") for a in sys.argv[1:]):
    CFG = {k: "" for k in ("bucket", "s3_root", "image_uri",
                           "sagemaker_role_arn")}
else:
    CFG = load_config()
PROFILE = CFG.get("profile")
BUCKET = CFG["bucket"]
S3 = CFG["s3_root"]
IMAGE = CFG["image_uri"]
ROLE = CFG["sagemaker_role_arn"]


def log(msg):
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    log("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def sm_client():
    return get_client("sagemaker", session=boto_session(CFG))


def channel(name, uri):
    return {"ChannelName": name,
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": uri,
                "S3DataDistributionType": "FullyReplicated"}},
            "InputMode": "File"}


def resolve_data_path(p):
    """Expand ~ and anchor relative paths at the repo root, so a
    project_config.json entry like "dataset/my_robot" means the repo's
    gitignored dataset/ folder no matter which cwd the pipeline runs from.
    (Unconditional on purpose: an exists()-from-cwd shortcut would silently
    retarget the same config value whenever the invoking directory happened
    to contain a same-named entry. CLI paths get cwd-first treatment where
    they are parsed, not here.)"""
    path = Path(os.path.expanduser(p))
    if path.is_absolute():
        return path.resolve()
    return (HERE.parent / path).resolve()


def wait_for_job(job, poll=120, tolerate_failed=None):
    # A multi-hour poll must survive credential rotation (STS tokens expire):
    # build a fresh client each iteration and retry auth/transient errors
    # instead of dying — the job itself is unaffected either way.
    import botocore.exceptions
    last, errs = None, 0
    while True:
        try:
            d = sm_client().describe_training_job(TrainingJobName=job)
            errs = 0
        except (botocore.exceptions.ClientError,
                botocore.exceptions.BotoCoreError) as e:
            # A bad job name is the one error retrying cannot fix — SageMaker
            # returns ValidationException for a name that does not exist, so a
            # typo would otherwise poll forever. Everything else is transient.
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code == "ValidationException":
                sys.exit(f"ABORT: SageMaker does not recognise job {job}: {e}")
            errs += 1
            # Retry indefinitely. There is deliberately no cap: the caller may
            # be babysitting a multi-day run, credentials here are refreshed by
            # hand (never automatically), and an expired token can therefore
            # outlast any bound. A cap of 30 x 60s once dropped a live 7000-step
            # run's poller after 30 min of ExpiredTokenException — the job kept
            # running server-side, so exiting bought nothing and lost the watch.
            if errs == 1 or errs % 10 == 0:
                log(f"  describe_training_job error ({errs}, retrying "
                    f"indefinitely): {e} — refresh credentials for profile "
                    f"{PROFILE or '(default)'} if this persists")
            time.sleep(min(poll, 60))
            continue
        st = (d["TrainingJobStatus"], d.get("SecondaryStatus", "-"))
        if st != last:
            log(f"  {job}: {st[0]}/{st[1]}")
            last = st
        if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            if d["TrainingJobStatus"] != "Completed":
                sec = d.get("SecondaryStatus", "-")
                # A caller may know that a given failure left usable artifacts
                # (see finalization_only_failure); it returns why to continue.
                if tolerate_failed:
                    ok = tolerate_failed(d)
                    if ok:
                        log(f"  {job} ended {d['TrainingJobStatus']}/{sec}, but "
                            f"{ok}")
                        return d
                why = d.get("FailureReason")
                if not why and d["TrainingJobStatus"] == "Stopped":
                    # A Stopped job carries no FailureReason at all, so the only
                    # discriminator is SecondaryStatus. Two very different causes
                    # land here and the remedies are opposite, so don't guess:
                    # our own StoppingCondition timing out, versus someone
                    # calling StopTrainingJob from outside.
                    if sec in ("MaxRuntimeExceeded", "MaxWaitTimeExceeded"):
                        why = (f"{sec} — this pipeline's own StoppingCondition "
                               f"fired, nothing external stopped the job. Raise "
                               f"max_runtime_hours for this stage in "
                               f"project_config.json and resume with --start-at")
                    else:
                        why = ("no FailureReason — the job was stopped via the "
                               "API, not by a training error and not by our "
                               "runtime cap. If it never left Pending, check "
                               "CloudTrail for StopTrainingJob to see which "
                               "principal stopped it (account cost-cleanup "
                               "tooling is a common cause; see extra_job_tags)")
                sys.exit(f"ABORT: {job} ended {d['TrainingJobStatus']}/{sec}: "
                         f"{why}")
            return d
        time.sleep(poll)


def launch_training(name, dataset_s3, fps, max_steps, save_steps, compute,
                    hp, dry, recipe="yam", spot=False):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"dz-{name}-{max_steps}s-{stamp}"[:63]
    spec = {
        "TrainingJobName": job,
        "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        "RoleArn": ROLE,
        "InputDataConfig": [
            channel("dataset", dataset_s3),
            channel("wan", f"{S3}/checkpoints/Wan2.1-I2V-14B-480P/"),
            channel("tokenizer", f"{S3}/checkpoints/umt5-xxl-tokenizer/"),
            channel("agibot", f"{S3}/checkpoints/DreamZero-AgiBot/"),
        ],
        "OutputDataConfig": {"S3OutputPath": f"{S3}/output/"},
        # Deliberately NO CheckpointConfig. SageMaker's checkpoint sync agent
        # can fail the whole job with a generic InternalServerError whenever a
        # multi-GB save burst lands in its LocalPath — observed both at a
        # mid-run checkpoint save (killing an otherwise healthy multi-day run
        # at its first save) and at teardown after training succeeded, 4/4 in
        # this account (eu-central-1, ml.g7e.48xlarge, 2026-08-18/19), while a
        # byte-identical job without CheckpointConfig Completed.
        #
        # AWS documents the same symptom and prescribes this workaround:
        # https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-troubleshooting-model-parallel.html
        # § "Saving Checkpoints". Quote it honestly: AWS says the error "*could*
        # be caused by a SageMaker AI limitation while uploading the local
        # checkpoint to Amazon S3 during training", and that page is archived
        # SMP-v1 material whose remedy snippet gates on smp.local_rank(),
        # whereas this recipe uses DeepSpeed ZeRO-2 and never SMP. So the doc
        # is a matching symptom plus a sanctioned workaround, not a ruling on
        # this configuration — the 4/4 above is the actual evidence, and no
        # published size limit exists to appeal to.
        #
        # The entrypoint (image v9+) owns BOTH directions of the sync: same
        # checkpoints-sync/ layout — see checkpoint_s3_uri below — a 60s upload
        # loop for live loss streaming and crash survival, a verified final
        # sync BEFORE exit, and (image v10+) a restore of the newest complete
        # checkpoint at start-up so a managed-spot relaunch resumes instead of
        # silently retraining from step 0 over its own recovery point.
        "ResourceConfig": {"InstanceType": compute["instance_type"],
                           "InstanceCount": 1,
                           "VolumeSizeInGB": int(compute["volume_gb"])},
        "StoppingCondition": {
            "MaxRuntimeInSeconds": int(float(compute["max_runtime_hours"]) * 3600)},
        "HyperParameters": {
            "max_steps": str(max_steps), "save_steps": str(save_steps),
            "learning_rate": str(hp["learning_rate"]),
            "per_device_train_batch_size": str(hp["per_device_train_batch_size"]),
            "seed": str(hp["seed"]), "warmup_ratio": str(hp["warmup_ratio"]),
            # fps flows from the embodiment config (v3 image parameterizes it)
            "fps_yam": str(fps),
            # which upstream data recipe the entrypoint runs: "yam" (GEAR/yam
            # bimanual, the default) or "droid" (single-arm Franka DROID)
            "recipe": recipe,
            # where the entrypoint's own sync loop mirrors /opt/ml checkpoints
            # (60s cadence + verified final sync). Same layout CheckpointConfig
            # used to produce, so scan_synced_checkpoints and the merge
            # recovery path read it unchanged. Requires image v9+.
            "checkpoint_s3_uri": f"{S3}/checkpoints-sync/{job}/",
        },
        "Tags": job_tags(CFG),
    }
    if spot:
        # Managed spot is OPT-IN (--spot), never the default. Two reasons the
        # default stays on-demand: a reclaim can leave the job queued for hours
        # (measured: 6.5 h of "Insufficient capacity error from EC2 while
        # launching instances, retrying!" after a real p5en reclaim), which reads
        # as a broken sample to a first-time user; and the savings only pay off
        # over a long run, not over the 1000-step demo.
        #
        # MaxWaitTimeInSeconds is REQUIRED with EnableManagedSpotTraining and
        # must be >= MaxRuntimeInSeconds — it caps runtime PLUS all the waiting.
        # 2x runtime with a 30 min floor gives a reclaimed job room to queue for
        # capacity and still finish; SageMaker stops the job at this bound
        # whether or not it ever got hardware.
        run_s = spec["StoppingCondition"]["MaxRuntimeInSeconds"]
        spec["EnableManagedSpotTraining"] = True
        spec["StoppingCondition"]["MaxWaitTimeInSeconds"] = max(run_s * 2, run_s + 1800)
    # Optional passthroughs from project_config.json "training":
    #   extra_overrides         — raw hydra overrides appended to the trainer CLI
    #   dataloader_num_workers  — loader parallelism (entrypoint default: 8)
    for opt in ("extra_overrides", "dataloader_num_workers"):
        if hp.get(opt) not in (None, ""):
            spec["HyperParameters"][opt] = str(hp[opt])
    if dry:
        log(f"  DRY-RUN spec for {job}:")
        print(json.dumps(spec, indent=2))
        return job
    sm_client().create_training_job(**spec)
    log(f"  submitted {job}")
    return job


def launch_merge(name, lora_job, compute, dry):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"dz-{name}-merge-{stamp}"[:63]
    merged_prefix = f"{S3}/models/{name}-merged"
    # The merged prefix is keyed on --name only, and merge_lora.py uploads
    # object-by-object with no delete onto an unversioned bucket — so a second
    # merge under the same name would overwrite ~92GB of servable weights in
    # place, with a mid-upload window where the prefix holds a mix of old and
    # new shards under a byte-identical index.json. Refuse instead of
    # overwriting; a fresh --name (or an explicit cleanup) is the safe path.
    if not dry:
        s3cli = get_client("s3", session=boto_session(CFG))
        existing = s3cli.list_objects_v2(
            Bucket=BUCKET, Prefix=f"{s3_root_key()}/models/{name}-merged/",
            MaxKeys=1)
        if existing.get("KeyCount"):
            sys.exit(
                f"ABORT: {merged_prefix}/ already holds objects — refusing to "
                f"overwrite servable weights in place (a partial upload would "
                f"leave a mixed old/new shard set). Either re-run with a new "
                f"--name, or delete the old artifact first:\n"
                f"  aws s3 rm --recursive {merged_prefix}/ --dryrun")
    spec = dict(
        TrainingJobName=job,
        AlgorithmSpecification={
            "TrainingImage": IMAGE, "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "/opt/ml/input/data/lora/merge_lora.py"]},
        RoleArn=ROLE,
        InputDataConfig=[
            channel("agibot", f"{S3}/checkpoints/DreamZero-AgiBot/"),
            channel("wan", f"{S3}/checkpoints/Wan2.1-I2V-14B-480P/"),
            channel("tokenizer", f"{S3}/checkpoints/umt5-xxl-tokenizer/"),
            channel("lora", f"{S3}/lora-checkpoints/{lora_job}/"),
        ],
        OutputDataConfig={"S3OutputPath": f"{S3}/output/"},
        ResourceConfig={"InstanceType": compute["instance_type"],
                        "InstanceCount": 1,
                        "VolumeSizeInGB": int(compute["volume_gb"])},
        StoppingCondition={
            "MaxRuntimeInSeconds": int(float(compute["max_runtime_hours"]) * 3600)},
        Environment={"LORA_DIR": "/opt/ml/input/data/lora",
                     "AGIBOT_DIR": "/opt/ml/input/data/agibot",
                     "MERGED_DIR": "/opt/ml/input/data/merged",
                     "MERGED_S3_URI": merged_prefix,
                     # merge_lora.py runs in the container as a single uploaded
                     # file, with no repo to import solution.py from, so the
                     # solution user-agent suffix has to be injected here — it
                     # is the only SDK caller inside any of our jobs (the
                     # train/eval containers only shell out to the aws CLI,
                     # which has no user-agent hook at all).
                     "USER_AGENT_STRING": user_agent_string()},
        Tags=job_tags(CFG, "merge"),
    )
    if dry:
        log(f"  DRY-RUN spec for {job}:")
        print(json.dumps(spec, indent=2))
        return job, merged_prefix
    sm_client().create_training_job(**spec)
    log(f"  submitted {job}")
    return job, merged_prefix


def s3_root_key():
    """The s3_root as a key prefix inside BUCKET (no scheme, no trailing slash)."""
    if not S3.startswith(f"s3://{BUCKET}/"):
        sys.exit(f"ABORT: s3_root {S3} is not inside bucket {BUCKET}")
    return S3[len(f"s3://{BUCKET}/"):].rstrip("/")


def scan_synced_checkpoints(train_job, s3cli, root_key):
    """Enumerate what actually reached checkpoints-sync/ for a training job.

    The entrypoint's sync loop writes this prefix (legacy jobs launched with
    CheckpointConfig share the exact same layout).

    Returns (complete, weights, cfg_files, root_files). `complete` is the sorted
    list of steps whose checkpoint-<N>/ holds BOTH config.json and
    model.safetensors — the two files merge_lora.py loads — so it is the
    authoritative answer to "which checkpoints are mergeable".
    """
    sync = f"{root_key}/checkpoints-sync/{train_job}"
    pages = s3cli.get_paginator("list_objects_v2")
    steps, cfg_files, root_files = {}, set(), {}
    for page in pages.paginate(Bucket=BUCKET, Prefix=f"{sync}/"):
        for o in page.get("Contents", []):
            rel = o["Key"][len(sync) + 1:]
            m = re.match(r"checkpoint-(\d+)/(config\.json|model\.safetensors)$", rel)
            if m:
                steps.setdefault(int(m.group(1)), {})[m.group(2)] = o["Key"]
            elif rel.startswith("experiment_cfg/"):
                cfg_files.add(rel)
            elif rel in ("loss_log.jsonl", "wandb_config.json"):
                root_files[rel] = o["Key"]
    complete = sorted(s for s, f in steps.items()
                      if {"config.json", "model.safetensors"} <= set(f))
    return complete, steps, cfg_files, root_files


def finalization_only_failure(train_job, s3cli, max_steps):
    """Build a wait_for_job() predicate that tolerates a post-training failure.

    SageMaker can fail a job *after* the container has exited 0 (see
    pipeline/README.md). Root cause was CheckpointConfig's sync agent choking
    on a multi-GB save burst; launch_training no longer sets it, so this
    should not fire on image v9+ jobs — it stays as belt-and-suspenders and
    for jobs launched by older revisions. No model.tar.gz is written in
    that failure mode, but the LoRA weights are already in checkpoints-sync/
    and stage_lora_for_merge recovers them — so the pipeline should continue to
    merge rather than drop a finished multi-day run on the floor. (On a legacy
    job the same agent could also kill the run at a MID-RUN save; that case
    correctly aborts here — checkpoint-<max_steps> is absent — and the log
    prints which earlier checkpoints are mergeable.)

    The discriminator is the *final* checkpoint, not merely "some checkpoint":
    a job that died at step 3000 also leaves complete earlier checkpoints, and
    silently merging a half-trained model is far worse than stopping. Only
    checkpoint-<max_steps> proves training ran all the way to its last save.
    """
    def check(d):
        if d["TrainingJobStatus"] != "Failed":
            # Stopped means our runtime cap fired or someone called
            # StopTrainingJob; neither implies training finished.
            return None
        complete, _, cfg_files, root_files = scan_synced_checkpoints(
            train_job, s3cli, s3_root_key())
        if max_steps not in complete:
            log(f"  no complete checkpoint-{max_steps} in checkpoints-sync "
                f"(mergeable checkpoints: {complete or 'none'}) — training did "
                f"not reach its final save, so this is a real failure")
            return None
        if not cfg_files or "loss_log.jsonl" not in root_files:
            log(f"  checkpoint-{max_steps} is complete but experiment_cfg/ or "
                f"loss_log.jsonl did not sync — merge would abort on those")
            return None
        return (f"checkpoint-{max_steps} (the final save) is complete in "
                f"checkpoints-sync, so training finished and only SageMaker's "
                f"post-container finalization failed — continuing to merge")
    return check


def stage_lora_from_checkpoint_sync(train_job, s3cli, root_key):
    """Fallback: build lora-checkpoints/ from the last synced checkpoint.

    merge_lora.py needs config.json, model.safetensors, experiment_cfg/ and
    loss_log.jsonl, and all four are in checkpoints-sync/<job>/ independently of
    model.tar.gz. The trainer writes a checkpoint's LoRA weights *before* its
    DeepSpeed resume states, so when a mid-run death (or, on legacy
    CheckpointConfig jobs, a finalization failure) truncates the sync it
    is the resume states that are lost, not these. Server-side copies only: the
    safetensors is a few hundred MB and never needs to touch this machine.

    loss_log.jsonl is only provenance, but merge_lora.py copies it *after* writing
    the 92GB merged checkpoint — so omitting it would waste a whole merge job.
    """
    sync = f"{root_key}/checkpoints-sync/{train_job}"
    complete, steps, have_root_cfg, root_files = scan_synced_checkpoints(
        train_job, s3cli, root_key)
    if not complete:
        sys.exit(f"ABORT: no checkpoint-<N>/ with both config.json and "
                 f"model.safetensors under s3://{BUCKET}/{sync}/ — nothing to merge")
    if not have_root_cfg:
        sys.exit(f"ABORT: experiment_cfg/ missing under s3://{BUCKET}/{sync}/ — "
                 f"merge_lora.py cannot build the inference config without it")
    if "loss_log.jsonl" not in root_files:
        sys.exit(f"ABORT: loss_log.jsonl missing under s3://{BUCKET}/{sync}/ — "
                 f"merge_lora.py copies it after the 92GB save, so a merge job "
                 f"would run for ~40min and then die before uploading anything")
    step = complete[-1]
    log(f"  no model.tar.gz; recovering LoRA from checkpoint-{step} "
        f"(complete checkpoints in S3: {complete})")
    dst = f"{root_key}/lora-checkpoints/{train_job}"
    for src in steps[step].values():
        s3cli.copy_object(Bucket=BUCKET, CopySource={"Bucket": BUCKET, "Key": src},
                          Key=f"{dst}/{Path(src).name}")
    for rel in sorted(have_root_cfg) + sorted(root_files):
        s3cli.copy_object(Bucket=BUCKET,
                          CopySource={"Bucket": BUCKET, "Key": f"{sync}/{rel}"},
                          Key=f"{dst}/{rel}")
    return step


def stage_lora_for_merge(train_job, s3cli):
    """Copy the training job's final artifacts + merge script to lora-checkpoints/.

    The normal source is the training job's model.tar.gz, but SageMaker writes that
    only for a job that reaches Completed — and a job can fail in *finalization*
    after the training itself fully succeeded (see pipeline/README.md caveats). In
    that case fall back to the synced checkpoints, which carry the same weights.
    """
    import tarfile
    import tempfile
    root_key = s3_root_key()
    tar_key = f"{root_key}/output/{train_job}/output/model.tar.gz"
    try:
        s3cli.head_object(Bucket=BUCKET, Key=tar_key)
    except s3cli.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "403"):
            raise
        stage_lora_from_checkpoint_sync(train_job, s3cli, root_key)
        s3cli.upload_file(str(HERE / "merge_lora.py"), BUCKET,
                          f"{root_key}/lora-checkpoints/{train_job}/merge_lora.py")
        return
    with tempfile.TemporaryDirectory() as td:
        tar = f"{td}/model.tar.gz"
        s3cli.download_file(BUCKET, f"{root_key}/output/{train_job}/output/model.tar.gz", tar)
        with tarfile.open(tar) as t:
            try:
                # nosec B202 — extraction is traversal-safe on both paths:
                # filter="data" rejects ../, links and devices, and the
                # TypeError fallback below validates every member manually.
                t.extractall(td, filter="data")  # nosec B202
            except TypeError:
                # Python without the filter= backport: validate members manually
                base = Path(td).resolve()
                for m in t.getmembers():
                    if not m.isfile() and not m.isdir():
                        sys.exit(f"ABORT: non-file member in model.tar.gz "
                                 f"(link/device): {m.name}")
                    if not (base / m.name).resolve().is_relative_to(base):
                        sys.exit(f"ABORT: unsafe path in model.tar.gz: {m.name}")
                t.extractall(td)  # nosec B202 — members validated above
        Path(f"{td}/model.tar.gz").unlink()
        for junk in Path(td).rglob("*.sagemaker-uploaded"):
            junk.unlink()
        for p in Path(td).rglob("*"):
            if p.is_file():
                key = f"{root_key}/lora-checkpoints/{train_job}/{p.relative_to(td)}"
                s3cli.upload_file(str(p), BUCKET, key)
        s3cli.upload_file(str(HERE / "merge_lora.py"), BUCKET,
                          f"{root_key}/lora-checkpoints/{train_job}/merge_lora.py")


def fetch_dataset(proj_ds, name):
    """Download a HuggingFace dataset repo to the local cache dir."""
    import shutil
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    dest = resolve_data_path(proj_ds["cache_dir"]) / name
    # snapshot_download never removes files, so reusing a --name for a
    # different repo/revision would leave a union of both trees in dest
    marker = dest / ".dz_source"
    source = f"{proj_ds['repo_id']}@{proj_ds['revision'] or 'default'}"
    if dest.exists() and not marker.exists() and (
            not dest.is_dir() or any(dest.iterdir())):
        # downloads and hand-placed local datasets share the dataset/ folder;
        # only the marker distinguishes them. Downloading into a user's
        # dataset would silently merge two trees (and a later run could
        # rmtree it) — refuse instead of guessing.
        sys.exit(f"ABORT: {dest} already exists but was not downloaded by "
                 f"this pipeline (no .dz_source marker) — refusing to "
                 f"download {source} into it. Pick a different --name, or "
                 f"move/delete that folder if it really is a stale download")
    if dest.exists() and marker.exists() and marker.read_text() != source:
        log(f"cache {dest} holds {marker.read_text()} — clearing for {source}")
        shutil.rmtree(dest)
    log(f"fetching {proj_ds['repo_id']} "
        f"(revision={proj_ds['revision'] or 'default'}) -> {dest}")
    # marker BEFORE the download: hf_hub writes into dest immediately, so an
    # interrupted run would otherwise leave a non-empty unmarked dir that the
    # guard above bricks — with the marker down, a plain retry resumes it
    dest.mkdir(parents=True, exist_ok=True)
    marker.write_text(source)
    snapshot_download(repo_id=proj_ds["repo_id"], repo_type="dataset",
                      revision=proj_ds["revision"], local_dir=str(dest),
                      max_workers=16)  # nosec B615 — revision comes from
    # project_config.json (the shipped default pins the v2.1 tag); a user
    # choosing revision=null is deliberately choosing the repo's default branch
    return dest


def preflight(args, proj, start, stop_idx):
    """Fail-fast status report before any stage runs.

    Everything here is a check the pipeline would otherwise discover late and
    expensively: a missing image or unstaged base weights fail a $20+/hr
    training job at download time, a quota of 0 is a synchronous
    ResourceLimitExceeded after an hour of dataset prep, and a missing
    dreamzero clone kills prep halfway through a conversion. All checks are
    read-only and stage-aware — only what the requested range actually needs
    is checked; the rest reports as skipped.
    """
    def need(*names):
        return any(start <= STAGES.index(n) <= stop_idx for n in names
                   if not (n == "smoke" and args.skip_smoke))

    problems = []

    def row(status, label, detail):
        mark = {"ok": "ok  ", "fail": "FAIL", "skip": "--  ", "warn": "??  "}[status]
        print(f"    {mark} {label:<16} {detail}", flush=True)
        if status == "fail":
            problems.append(f"{label}: {detail}")

    log("preflight:")
    sess = boto_session(CFG)

    # credentials + bucket: everything from `stage` onward talks to AWS
    if need("stage", "smoke", "train", "merge"):
        try:
            ident = get_client("sts", session=sess).get_caller_identity()
            row("ok", "credentials", f"account {ident['Account']}")
        except Exception as e:
            row("fail", "credentials", f"{type(e).__name__} — refresh AWS "
                                       f"credentials for profile "
                                       f"{PROFILE or '(default)'}")
            ident = None
        try:
            get_client("s3", session=sess).head_bucket(Bucket=BUCKET)
            row("ok", "s3 bucket", BUCKET)
        except Exception as e:
            row("fail", "s3 bucket", f"{BUCKET}: {type(e).__name__} — wrong "
                                     f"account/region, or the CDK stack is not "
                                     f"deployed")
    else:
        row("skip", "credentials", "no AWS stage in the requested range")

    # container image: the smoke/train/merge jobs all run it
    if need("smoke", "train", "merge"):
        # image_uri may be tag-addressed (repo:tag) or digest-pinned
        # (repo@sha256:...) — generate_pipeline_config.py emits either
        path = IMAGE.split("/", 1)[1]
        if "@" in path:
            repo_name, _, ref = path.partition("@")
            image_id, shown = {"imageDigest": ref}, f"@{ref[7:19]}…"
        else:
            repo_name, _, ref = path.partition(":")
            image_id, shown = {"imageTag": ref}, f":{ref}"
        try:
            img = get_client("ecr", session=sess).describe_images(
                repositoryName=repo_name, imageIds=[image_id]
            )["imageDetails"][0]
            # Launcher/image pairing: this launcher relies on the v9+
            # entrypoint syncing checkpoints itself (checkpoint_s3_uri). A
            # pre-v9 image would write checkpoints to the unmounted
            # /opt/ml/checkpoints — the small root overlay — and die with
            # ENOSPC at the first multi-GB save, hours in. Tags are the only
            # version signal ECR has; a digest-pinned URI still resolves to
            # its tags here.
            vers = [int(m.group(1)) for t in (img.get("imageTags") or [])
                    if (m := re.fullmatch(r"v(\d+)", t))]
            if vers and max(vers) < 9:
                row("fail", "training image",
                    f"{repo_name}{shown} is v{max(vers)}, but this launcher "
                    f"needs image v9+ (the entrypoint owns checkpoint sync; "
                    f"a pre-v9 image fills the root overlay and dies with "
                    f"ENOSPC mid-run) — build/push v11 and update image_uri")
            elif getattr(args, "spot", False) and vers and max(vers) < 11:
                # --spot turns the version advisory into a hard gate. On demand a
                # missing restore is unreachable (SageMaker never relaunches an
                # on-demand job); with managed spot it is the difference between
                # resuming and silently overwriting the recovery point you would
                # have resumed from, in a bucket with no versioning.
                row("fail", "training image",
                    f"{repo_name}{shown} is v{max(vers)}, and --spot needs v11+. "
                    f"A reclaim relaunches this same job spec into the same "
                    f"checkpoint prefix: v9 has no restore at all, and v10 has "
                    f"three holes (a staged ckpt channel reports SUCCESS having "
                    f"trained nothing, a bucket-root URI never finds its mirror, "
                    f"an all-rejected prefix looks empty). Build/push v11 and "
                    f"update image_uri, or drop --spot")
            elif vers and max(vers) < 11:
                # Not fatal for an on-demand run: SageMaker never relaunches
                # one, so the restore path is unreachable either way. It matters
                # for managed spot, which this launcher does not use but ad-hoc
                # submitters do. v9 is the dangerous one — a relaunch retrains
                # from step 0 AND re-uploads over its own checkpoint keys in a
                # non-versioned bucket, destroying the recovery point (measured
                # 2026-08-30: ~3,900 steps / ~$480 lost). v10 restores, but
                # carries three holes v11 closes: a `ckpt` channel staged from a
                # previous run's output prefix makes the job report SUCCESS
                # having trained nothing, a bucket-root checkpoint URI never
                # finds its own mirror, and an all-rejected prefix is logged as
                # if it were empty.
                row("warn", "training image",
                    f"{repo_name}{shown} is v{max(vers)} — usable on demand, but "
                    f"for managed spot prefer v11+: v10+ restores the newest "
                    f"complete checkpoint at start-up so a relaunch resumes "
                    f"instead of retraining from step 0 over its own recovery "
                    f"point, and v11 fixes three ways that restore can still "
                    f"silently do the wrong thing"
                    + (" — v9 has no restore at all, do NOT spot it"
                       if max(vers) < 10 else ""))
            else:
                row("ok", "training image",
                    f"{repo_name}{shown} ({img['imageSizeInBytes'] / 2**30:.1f} GiB, "
                    f"pushed {img['imagePushedAt']:%Y-%m-%d})")
        except Exception:
            row("fail", "training image",
                f"{repo_name}{shown} not in ECR — build and push it: "
                f"bash docker/build_and_push.sh <tag>, then update image_uri "
                f"in pipeline/pipeline_config.json")

        from stage_base_assets import check_staged
        try:
            if check_staged(CFG, quiet=True):
                row("ok", "base weights", "all three checkpoints staged in S3")
            else:
                row("fail", "base weights",
                    "incomplete — inspect: python3 pipeline/stage_base_assets.py "
                    "--check   stage: ./setup.sh --stage-assets")
        except Exception as e:
            row("warn", "base weights", f"could not check ({type(e).__name__})")

        # quota 0 is a synchronous rejection at submit time; catch it now.
        itypes = sorted({proj["compute"][s]["instance_type"]
                         for s in ("smoke", "train", "merge") if need(s)})
        wanted = {f"{t} for training job usage": t for t in itypes}
        try:
            found = {}
            pages = get_client("service-quotas", session=sess).get_paginator(
                "list_service_quotas")
            for page in pages.paginate(ServiceCode="sagemaker"):
                for q in page["Quotas"]:
                    if q["QuotaName"] in wanted:
                        found[wanted[q["QuotaName"]]] = q["Value"]
                if len(found) == len(wanted):
                    break  # ~1800 quotas; stop paging once all are answered
            for t in itypes:
                v = found.get(t)
                if v:
                    row("ok", "gpu quota", f"{t} = {v:g}")
                else:
                    row("fail", "gpu quota",
                        f"{t} training-job quota is {0 if v == 0 else 'absent'} "
                        f"in this region — request an increase (README "
                        f"prerequisites) before submitting")
        except Exception as e:
            row("warn", "gpu quota", f"could not check ({type(e).__name__}) — "
                                     f"submission will fail synchronously if 0")

    # the prep stage shells into the DreamZero repo's converter
    if need("prep"):
        dz = Path(os.environ.get("DREAMZERO_REPO", "./dreamzero")).resolve()
        if (dz / "scripts" / "data" / "convert_lerobot_to_gear.py").exists():
            row("ok", "dreamzero clone", str(dz))
        else:
            row("fail", "dreamzero clone",
                f"converter not found under {dz} — run: ./setup.sh")

    if need("validate", "prep"):
        import importlib.util
        missing = [m for m in ("pyarrow", "pandas", "imageio_ffmpeg")
                   if importlib.util.find_spec(m) is None]
        if missing:
            row("fail", "python deps", f"missing {', '.join(missing)} — run: "
                                       f"./setup.sh")
        else:
            row("ok", "python deps", "pyarrow, pandas, imageio-ffmpeg")

    if problems:
        if args.dry_run:
            log(f"preflight found {len(problems)} problem(s) — continuing "
                f"(dry-run)")
        elif args.skip_preflight:
            log(f"preflight found {len(problems)} problem(s) — continuing "
                f"(--skip-preflight)")
        else:
            sys.exit(f"ABORT: preflight found {len(problems)} problem(s) "
                     f"above. Fix them, or pass --skip-preflight if you are "
                     f"sure they are stale.")
    else:
        log("preflight: all checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="local LeRobot dataset dir (v2.1 or v3); overrides "
                         "project_config.json dataset settings")
    ap.add_argument("--hf-dataset", default=None, metavar="REPO_ID",
                    help="HuggingFace dataset repo id, e.g. "
                         "lerobot/aloha_static_screw_driver; overrides "
                         "project_config.json dataset settings")
    ap.add_argument("--hf-revision", default=None,
                    help="branch/tag/commit for --hf-dataset")
    ap.add_argument("--name", required=True,
                    help="label for this run: names the S3 prefixes "
                         "(datasets/<name>/, models/<name>-merged/), the "
                         "SageMaker jobs, and — for HuggingFace sources — the "
                         "download folder dataset/<name>/. Independent of any "
                         "local dataset folder name")
    ap.add_argument("--config", default=None,
                    help="embodiment config yaml (default: "
                         "project_config.json embodiment_config)")
    ap.add_argument("--project-config", default=None,
                    help="path to project_config.json (default: the repo "
                         "root's project_config.json)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--train-job", default=None,
                    help="an existing training job name. With --start-at merge, "
                         "merges that finished run without re-training; with "
                         "--start-at train, attaches to a job that is still "
                         "running (wait, then merge) instead of submitting a new "
                         "one — use this to reattach after a lost session")
    ap.add_argument("--spot", action="store_true",
                    help="run the TRAIN stage on managed spot (not the smoke "
                         "gate or the merge). Sets EnableManagedSpotTraining "
                         "and MaxWaitTimeInSeconds. Needs image v11+: on a "
                         "reclaim SageMaker relaunches the same job spec, and "
                         "only v11+ restores the newest complete checkpoint at "
                         "start-up instead of retraining from step 0 over its "
                         "own recovery point. Expect queueing — a reclaimed job "
                         "can sit in Starting for hours waiting for capacity, "
                         "billed nothing while it waits")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-validation", action="store_true",
                    help="skip the dataset-vs-config validation stage (not recommended)")
    ap.add_argument("--start-at", choices=STAGES, default="fetch")
    ap.add_argument("--stop-after", choices=STAGES, default=None,
                    help="exit cleanly after this stage completes — e.g. "
                         "--stop-after smoke to inspect smoke-job timing/loss "
                         "before committing to the full run (resume with "
                         "--start-at train)")
    ap.add_argument("--v3-converter", default=None,
                    help="command template for v3->v2.1, e.g. 'python3 "
                         "pipeline/convert_lerobot_v3_to_v21.py {src} {dst}' "
                         "(v3 layouts are dataset-specific; that shipped script "
                         "is a worked example, not a universal converter)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not abort on the read-only prerequisite checks "
                         "(image in ECR, base weights in S3, GPU quota, "
                         "dreamzero clone). They still run and still report — "
                         "this only downgrades a failure to a warning, for "
                         "when you know a finding is stale")
    ap.add_argument("--dry-run", action="store_true",
                    help="run local stages, print job specs, submit nothing")
    args = ap.parse_args()

    # --name is embedded in SageMaker training-job names, which allow only
    # alphanumerics and hyphens — reject a bad one here rather than at the
    # smoke job, after fetch/validate/prep have already run.
    if not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9])*", args.name):
        sys.exit(f"ABORT: --name {args.name!r} must be letters, digits and "
                 f"hyphens only (it becomes part of SageMaker job names)")

    proj = load_project_config(args.project_config)

    # CLI overrides for the dataset source. --dataset keeps normal CLI
    # semantics (relative to cwd) but falls back to the repo root so the
    # documented `--dataset dataset/my_robot` works from any directory.
    if args.dataset:
        d = Path(os.path.expanduser(args.dataset))
        if not d.is_absolute():
            alt = HERE.parent / d
            if not d.exists() and alt.exists():
                d = alt
            elif d.exists() and alt.exists() and d.resolve() != alt.resolve():
                log(f"note: --dataset {args.dataset} matches both the cwd and "
                    f"the repo root — using the cwd copy {d.resolve()}")
        proj["dataset"].update(source="local", local_path=str(d.resolve()))
    elif args.hf_dataset:
        # keep the configured revision only when it's the same repo; a
        # different repo's default branch is fetched unless --hf-revision says
        # otherwise (blindly reusing "v2.1" would 404 or fetch the wrong tree)
        rev = args.hf_revision or (proj["dataset"]["revision"]
                                   if args.hf_dataset == proj["dataset"]["repo_id"]
                                   else None)
        proj["dataset"].update(source="huggingface", repo_id=args.hf_dataset,
                               revision=rev)

    config_path = args.config or proj.get("embodiment_config")
    if not config_path:
        sys.exit("ABORT: no embodiment config — pass --config or set "
                 "embodiment_config in project_config.json")
    # relative config paths: cwd first, then pipeline/ ("configs/x.yaml"),
    # then the repo root ("pipeline/configs/x.yaml" works from any cwd)
    config_path = os.path.expanduser(config_path)
    if not Path(config_path).exists():
        for base in (HERE, HERE.parent):
            if (base / config_path).exists():
                config_path = str(base / config_path)
                break
        else:
            sys.exit(f"ABORT: embodiment config {config_path!r} not found "
                     f"(tried the current directory, {HERE} and "
                     f"{HERE.parent}; shipped configs live in "
                     f"pipeline/configs/)")
    cfg = yaml.safe_load(open(config_path))
    # format: lerobot (default) runs convert+prep; gear means the dataset
    # already ships GEAR metadata (modality.json + relative stats) and those
    # stages are skipped. recipe picks the trainer's data config.
    ds_format = cfg.get("format", "lerobot")
    recipe = cfg.get("recipe", "yam")

    max_steps = args.max_steps or int(proj["training"]["max_steps"])
    # under $HOME, not /tmp: the prepped dataset must survive across resumed
    # invocations (--start-at stage), and /tmp is world-writable/purgeable
    work = Path(os.path.expanduser("~/dreamzero_staging/work")) / args.name
    work.mkdir(parents=True, exist_ok=True)
    start = STAGES.index(args.start_at)
    stop_idx = (STAGES.index(args.stop_after) if args.stop_after
                else len(STAGES) - 1)
    preflight(args, proj, start, stop_idx)
    s3cli = get_client("s3", session=boto_session(CFG))
    dataset_s3 = f"{S3}/datasets/{args.name}/"

    def stop_after(stage):
        if args.stop_after == stage:
            log(f"--stop-after {stage}: stopping as requested "
                "(resume with --start-at)")
            sys.exit(0)

    # -- 1: fetch --
    if proj["dataset"]["source"] == "huggingface":
        if not proj["dataset"]["repo_id"]:
            sys.exit("ABORT: dataset.source is huggingface but no repo_id set "
                     "(project_config.json or --hf-dataset)")
        cached = resolve_data_path(proj["dataset"]["cache_dir"]) / args.name
        if start <= STAGES.index("fetch"):
            ds = fetch_dataset(proj["dataset"], args.name)
        else:
            ds = cached  # resumed run: reuse the earlier download
            # guard only the stages that read the local dataset (like the
            # meta/info.json check below): a merge-only resume must not
            # abort over a dataset it never touches
            marker = cached / ".dz_source"
            want = (f"{proj['dataset']['repo_id']}"
                    f"@{proj['dataset']['revision'] or 'default'}")
            if (start <= STAGES.index("prep") and marker.exists()
                    and marker.read_text() != want):
                sys.exit(f"ABORT: {cached} holds {marker.read_text()}, not "
                         f"{want} — pass the original --hf-dataset/"
                         f"--hf-revision, or rerun without --start-at to "
                         f"re-fetch")
    else:
        local = proj["dataset"]["local_path"]
        if not local:
            sys.exit("ABORT: no dataset — pass --dataset/--hf-dataset or set "
                     "dataset.repo_id/local_path in project_config.json")
        ds = resolve_data_path(local)
    # the local dataset is only dereferenced through the prep stage; a resume
    # from stage/smoke/train/merge must work without it (data already in S3)
    if start <= STAGES.index("prep") and not (ds / "meta" / "info.json").exists():
        sys.exit(f"ABORT: {ds} is not a LeRobot dataset (no meta/info.json)")
    stop_after("fetch")

    # -- 2/3: detect + convert --
    if ds_format == "gear" and start <= STAGES.index("convert"):
        # pre-prepped GEAR dataset: no version detection or conversion; just
        # verify the GEAR metadata the trainer dereferences is really there
        for req in ("modality.json", "relative_stats_dreamzero.json"):
            if not (ds / "meta" / req).exists():
                sys.exit(f"ABORT: config says format: gear but {ds}/meta/{req} "
                         "is missing — this is not a GEAR-prepped dataset")
        log("format=gear: skipping detect/convert (dataset is pre-prepped)")
    elif start <= STAGES.index("convert"):
        ver = json.load(open(ds / "meta" / "info.json")).get("codebase_version", "?")
        log(f"dataset codebase_version = {ver}")
        if ver != "v2.1":
            if not args.v3_converter:
                sys.exit(f"ABORT: dataset is {ver}; if it came from "
                         "HuggingFace, try the v2.1 tag first — most lerobot/* "
                         "repos still tag a v2.1 tree even though main now "
                         "points at v3: --hf-revision v2.1, or dataset.revision "
                         "in the project config. Otherwise convert it first with "
                         "pipeline/convert_lerobot_v3_to_v21.py (a worked example "
                         "for bimanual 3-camera v3 trees) and point "
                         "dataset.local_path at its output, or run it inline: "
                         "--v3-converter 'python3 "
                         "pipeline/convert_lerobot_v3_to_v21.py {src} {dst}'. "
                         "v3 layouts are dataset-specific, so a different tree "
                         "may need that script's split/cut logic adapted (see "
                         "pipeline/README.md Caveats)")
            out = work / "v21"
            run(args.v3_converter.format(src=ds, dst=out).split())
            ds = out
    elif (work / "v21" / "meta" / "info.json").exists():
        # resuming past convert on a v3 dataset: use the converted output,
        # not the raw source (validate/prep would otherwise see v3 files)
        ds = work / "v21"
        log(f"resume: using converted dataset at {ds}")
    stop_after("detect")
    stop_after("convert")

    # -- 4: validate (cheap; catches silent-garbage configs before any spend) --
    if (proj["validation"]["enabled"] and not args.skip_validation
            and start <= STAGES.index("validate")):
        vcmd = [sys.executable, str(HERE / "validate_dataset.py"),
                "--src", str(ds), "--config", config_path,
                "--sample-rows", str(proj["validation"]["degeneracy_sample_rows"])]
        if ds_format == "gear":
            vcmd.append("--gear")  # layout comes from meta/modality.json
        if proj["validation"]["fail_on_action_equals_state"]:
            vcmd.append("--fail-on-degeneracy")
        run(vcmd)
    stop_after("validate")

    # -- 5: prep --
    if ds_format == "gear":
        prepped = ds  # already GEAR: stage the dataset as-is
        if start <= STAGES.index("prep"):
            log("format=gear: skipping prep")
    else:
        prepped = work / "gear"
        if start <= STAGES.index("prep"):
            run([sys.executable, str(HERE / "prep_dataset.py"),
                 "--src", str(ds), "--dst", str(prepped), "--config", config_path])
    stop_after("prep")

    # -- 6: stage --
    if start <= STAGES.index("stage"):
        # --delete: the prefix must mirror this prep exactly — leftovers from
        # an earlier run with the same --name would train on mixed data.
        # Excludes keep hf-hub cache internals out of the training channel
        # when staging a pre-prepped (format: gear) dataset directly.
        # no --only-show-errors: let the CLI's live progress meter show
        sync_cmd = ["aws", "s3", "sync", str(prepped), dataset_s3,
                    "--delete",
                    "--exclude", ".cache/*", "--exclude", ".dz_source",
                    "--exclude", ".gitattributes"]
        if PROFILE:
            sync_cmd += ["--profile", PROFILE]
        run(sync_cmd)
        log(f"dataset staged: {dataset_s3}")
    stop_after("stage")

    # -- 7: smoke gate --
    if not args.skip_smoke and start <= STAGES.index("smoke"):
        smoke = proj["compute"]["smoke"]
        smoke_steps = int(smoke.get("steps", 10))
        # save_steps defaults to the step count (one save, at the end). Set
        # compute.smoke.save_steps lower to exercise MID-RUN checkpoint saves
        # in the gate — e.g. steps=30 + save_steps=10. Use a divisor of steps:
        # the gate below keys on checkpoint-<steps> being the final save.
        job = launch_training(args.name, dataset_s3, cfg["fps"],
                              max_steps=smoke_steps,
                              save_steps=int(smoke.get("save_steps",
                                                       smoke_steps)),
                              compute=smoke, hp=proj["training"],
                              dry=args.dry_run, recipe=recipe)
        if not args.dry_run:
            # The gate has to tolerate a finalization-only failure for the same
            # reason train does, and more urgently: this failure mode has hit
            # every job in some accounts, which would abort every fresh
            # end-to-end run at the gate even though the smoke passed.
            wait_for_job(job, tolerate_failed=finalization_only_failure(
                job, s3cli, smoke_steps))
            log("smoke gate PASSED")
    stop_after("smoke")

    # -- 8: train --
    train_job = args.train_job
    if start <= STAGES.index("train"):
        if train_job:
            # --train-job with --start-at train means "this run is already
            # submitted": wait for it and carry on. Never launch a second copy
            # of a multi-day job just because the local process was restarted.
            log(f"  attaching to existing training job {train_job}")
        else:
            # checkpoint at least every 500 steps (training.save_steps); never
            # let the interval exceed max_steps, or checkpoint-<max_steps> —
            # the final save the merge-recovery path keys on — is never written
            cfg_save = proj["training"].get("save_steps")
            save = (int(cfg_save) if cfg_save
                    else min(max(max_steps // 2, 10), 500))
            if save > max_steps:
                log(f"  save_steps {save} > max_steps {max_steps} — clamping "
                    f"to {max_steps} so the final checkpoint exists")
                save = max_steps
            train_job = launch_training(args.name, dataset_s3, cfg["fps"],
                                        max_steps=max_steps,
                                        save_steps=save,
                                        compute=proj["compute"]["train"],
                                        hp=proj["training"], dry=args.dry_run,
                                        recipe=recipe, spot=args.spot)
        if not args.dry_run:
            wait_for_job(train_job, tolerate_failed=finalization_only_failure(
                train_job, s3cli, max_steps))
    stop_after("train")

    # -- 9: merge --
    if start <= STAGES.index("merge"):
        if train_job is None:
            sys.exit("ABORT: --start-at merge needs the completed training job "
                     "name to locate the LoRA artifacts. Pass --train-job "
                     "<name> (from the earlier train stage's log line).")
        if not args.dry_run:
            stage_lora_for_merge(train_job, s3cli)
        job, merged = launch_merge(args.name, train_job,
                                   proj["compute"]["merge"], args.dry_run)
        if not args.dry_run:
            wait_for_job(job)
            log(f"DONE. Servable weights: {merged}/")
            log(f"       Raw LoRA (archival): {S3}/lora-checkpoints/{train_job}/")
        else:
            log(f"DRY-RUN complete. Would produce: {merged}/")


if __name__ == "__main__":
    main()
