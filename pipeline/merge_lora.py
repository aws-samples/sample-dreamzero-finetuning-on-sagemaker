#!/usr/bin/env python3
"""Merge the SageMaker LoRA checkpoint onto the DreamZero-AgiBot base.

Rebuilds the training-time composition (AgiBot full weights + our LoRA
deltas), merge_and_unload(), and saves a self-contained checkpoint that
inference can load WITHOUT hitting the repo's load_lora raw-Wan trap (that
naive path composes onto raw Wan2.1 and measured 9.9x worse MSE).

Run inside the training container (paths match the recorded config.json):
  docker run --rm \
    -v <staging>/checkpoints/Wan2.1-I2V-14B-480P:/opt/ml/input/data/wan:ro \
    -v <staging>/checkpoints/umt5-xxl:/opt/ml/input/data/tokenizer:ro \
    -v <staging>/checkpoints/DreamZero-AgiBot:/opt/ml/input/data/agibot:ro \
    -v <lora_ckpt_dir>:/opt/ml/lora:ro -v <out_dir>:/opt/ml/merged \
    --entrypoint python <image> /opt/ml/lora/merge_lora.py
"""
import gc
import json
import os
import shutil

import torch  # noqa: F401  (must import before groot)
from safetensors.torch import load_file

from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

CKPT = os.environ.get("LORA_DIR", "/opt/ml/lora")
AGIBOT = os.environ.get("AGIBOT_DIR", "/opt/ml/input/data/agibot")
OUT = os.environ.get("MERGED_DIR", "/opt/ml/merged")
# Upload directly from the script: a one-shot ~92GB burst into
# /opt/ml/checkpoints at job end overwhelms the CheckpointConfig sync agent
# (job dies with InternalServerError after "MERGE COMPLETE").
MERGED_S3_URI = os.environ.get("MERGED_S3_URI", "")

# config from OUR checkpoint, but with TRAINING-time flags
cfg = json.load(open(f"{CKPT}/config.json"))
inner = cfg["action_head_cfg"]["config"]
inner["defer_lora_injection"] = True     # inject LoRA AFTER the base loads
inner["skip_component_loading"] = True   # don't load raw-Wan DiT (AgiBot has it all)
model = VLA(VLAConfig(**cfg))
print("model constructed")

# 1. full AgiBot base, shard by shard (exactly like training)
index = json.load(open(f"{AGIBOT}/model.safetensors.index.json"))
for shard in sorted(set(index["weight_map"].values())):
    model.load_state_dict(load_file(f"{AGIBOT}/{shard}"), strict=False)
    gc.collect()
    print(f"loaded {shard}")

# 2. LoRA structure, then our trained deltas on top
model.action_head.inject_lora_after_loading()
missing, unexpected = model.load_state_dict(load_file(f"{CKPT}/model.safetensors"), strict=False)
print(f"LoRA overlay applied ({len(unexpected)} unexpected keys — expect 0)")
if unexpected:  # not assert: must survive python -O — a mismatched LoRA merged
    raise SystemExit(f"ABORT: unexpected keys: {unexpected[:5]}")  # silently is a broken model

# 3. merge adapters into the base, save full checkpoint
model.action_head.model = model.action_head.model.merge_and_unload()
inner["train_architecture"] = "full"
inner["defer_lora_injection"] = False
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
print("saved merged checkpoint")

# 4. experiment_cfg with inference-time flags so the full-load path is taken
shutil.copytree(f"{CKPT}/experiment_cfg", f"{OUT}/experiment_cfg", dirs_exist_ok=True)
conf = open(f"{OUT}/experiment_cfg/conf.yaml").read()
conf = conf.replace("save_lora_only: true", "save_lora_only: false")
conf = conf.replace("train_architecture: lora", "train_architecture: full")
open(f"{OUT}/experiment_cfg/conf.yaml", "w").write(conf)
# provenance only, and we are already past the expensive save — never let a
# missing side file throw away a finished 92GB merge
if os.path.exists(f"{CKPT}/loss_log.jsonl"):
    shutil.copy(f"{CKPT}/loss_log.jsonl", f"{OUT}/loss_log.jsonl")
else:
    print("note: no loss_log.jsonl in the LoRA dir; merged checkpoint is unaffected")
print("MERGE COMPLETE:", OUT)

if MERGED_S3_URI:
    import boto3
    from boto3.s3.transfer import TransferConfig
    # aliased: TransferConfig is passed to upload_file as Config= below, so the
    # bare name is already taken in this scope
    from botocore.config import Config as BotoConfig
    if not MERGED_S3_URI.startswith("s3://"):
        raise SystemExit(f"ABORT: MERGED_S3_URI must be s3://… (got {MERGED_S3_URI!r})")
    bucket, _, prefix = MERGED_S3_URI[5:].partition("/")
    prefix = prefix.rstrip("/")
    # This file is uploaded to the job as a standalone script, so there is no
    # repo alongside it to import pipeline/solution.py from — the solution
    # user-agent suffix is inlined instead, from the env var run_pipeline.py
    # sets on the job. .get so an unset var degrades to "no suffix" rather than
    # failing a merge that has already cost a training run.
    s3 = boto3.client("s3", config=BotoConfig(
        user_agent_extra=os.environ.get("USER_AGENT_STRING", "")))
    tc = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                        max_concurrency=16,
                        multipart_chunksize=64 * 1024 * 1024)
    for root, _, files in os.walk(OUT):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            key = f"{prefix}/{os.path.relpath(path, OUT)}"
            print(f"uploading {key} ({os.path.getsize(path)/1e9:.2f} GB)", flush=True)
            s3.upload_file(path, bucket, key, Config=tc)
    print("S3 UPLOAD COMPLETE:", MERGED_S3_URI)
