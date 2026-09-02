#!/bin/bash
# Open-loop evaluation of DreamZero checkpoints INSIDE a SageMaker job.
# The "training job" API is used purely as managed-GPU orchestration — this
# script only runs inference (torch.inference_mode; no gradients, no updates).
#
# Channels expected: one per model arm (see ARMS env), plus wan, tokenizer,
# dataset, evalscript. Results upload per arm to $RESULTS_S3_URI/<arm>/.
#
# Checkpoint-portability fixes baked in (each one cost us a failed job):
#   a) some configs name components by bare HF repo id, resolved relative to
#      the working directory — symlink those ids to the mounted channels
#   b) a checkpoint records frozen-component paths from ITS OWN training env
#      (or stores null); either way the repo's ensure_file() falls back to
#      downloading from the HuggingFace hub — a code-level fallback no symlink
#      can intercept, fatal offline. Repoint any unresolvable path to a channel.
#   c) a BASE checkpoint (never fine-tuned on your embodiment) lacks your
#      embodiment's transforms/stats entirely — evaluate it by composing your
#      fine-tuned checkpoint's config.json + experiment_cfg over the base's
#      weight shards (the same composition LoRA training itself uses)
set -uo pipefail
RES=${RESULTS_S3_URI:?set RESULTS_S3_URI}
FRAMES=${NUM_SAMPLES:-150}
EMB_TAG=${EMBODIMENT_TAG:-yam}   # yam (GEAR bimanual) | oxe_droid (DROID)

SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH=$(find "$SITE/nvidia" -maxdepth 2 -type d -name lib | paste -sd:):${LD_LIBRARY_PATH:-}
pip install --no-cache-dir --quiet awscli 2>/dev/null || true

cd /opt/ml/code/dreamzero
cp /opt/ml/input/data/evalscript/open_loop_eval.py scripts/

# --- fix (a): bare HF repo ids resolved relative to cwd -> channels ---
# Some configs store "Wan-AI/Wan2.1-I2V-14B-480P" / "google/umt5-xxl" rather
# than a path; the repo resolves those relative to the working directory.
mkdir -p Wan-AI google
ln -sfn /opt/ml/input/data/wan       Wan-AI/Wan2.1-I2V-14B-480P
ln -sfn /opt/ml/input/data/tokenizer google/umt5-xxl

# --- fix (b): repoint component paths in a model dir's config.json ---
# A checkpoint records the frozen-component paths from ITS OWN training
# environment, which do not exist here; some checkpoints store null instead.
# Both are fatal, because the repo's ensure_file() treats a null/missing path as
# "download from the HuggingFace hub" — a code-level fallback that no symlink
# can intercept, and that cannot work in an offline container. So rewrite every
# component path that is null OR does not resolve locally to its mounted
# channel. This is deliberately generic: it does not assume anything about
# where the checkpoint was originally trained.
patch_component_paths() {
  python - "$1" <<'PYEOF'
import json, os, sys
p = f"{sys.argv[1]}/config.json"
if not os.path.exists(p):
    sys.exit(0)
cfg = json.load(open(p))
inner = cfg.get("action_head_cfg", {}).get("config", {})
WAN = "/opt/ml/input/data/wan"
fixes = {
  ("text_encoder_cfg", "text_encoder_pretrained_path"):
      f"{WAN}/models_t5_umt5-xxl-enc-bf16.pth",
  ("image_encoder_cfg", "image_encoder_pretrained_path"):
      f"{WAN}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
  ("vae_cfg", "vae_pretrained_path"): f"{WAN}/Wan2.1_VAE.pth",
}
changed = []
for (sub, key), val in fixes.items():
    if not isinstance(inner.get(sub), dict):
        continue
    cur = inner[sub].get(key)
    if cur in (None, "") or not os.path.exists(cur):
        inner[sub][key] = val
        changed.append(f"{sub}.{key}: {cur!r} -> {val}")
if changed:
    json.dump(cfg, open(p, "w"), indent=2)
    print(f"repointed component paths in {p}:")
    for c in changed:
        print(f"  {c}")
PYEOF
}

# --- fix (c): base-arm composition (opt-in via BASE_CONFIG_DONOR channel) ---
prepare_base_arm() {  # <base_model_dir> -> echoes prepared dir
  local SRC=$1 DONOR=${BASE_CONFIG_DONOR:-}
  if [ -z "$DONOR" ] || [ ! -d "$DONOR" ]; then echo "$SRC"; return; fi
  local DST=/tmp/base_composed
  rm -rf $DST && mkdir -p $DST
  ln -s "$SRC"/*.safetensors $DST/ 2>/dev/null
  cp "$SRC"/model.safetensors.index.json $DST/
  cp "$DONOR"/config.json $DST/
  cp -r "$DONOR"/experiment_cfg $DST/
  echo "$DST"
}

FAIL=0
run_arm() {  # <name> <model_dir> [is_base]
  local NAME=$1 DIR=$2
  if [ "${3:-}" = "base" ]; then DIR=$(prepare_base_arm "$DIR"); fi
  echo "===== ARM $NAME ($DIR) ====="
  patch_component_paths "$DIR" || true
  mkdir -p /tmp/results_$NAME
  python scripts/open_loop_eval.py \
      --model_path "$DIR" \
      --dataset_path /opt/ml/input/data/dataset \
      --device cuda:0 --num_samples "$FRAMES" --spread_samples \
      --use_dataset_prompt --embodiment_tag "$EMB_TAG" \
      --output_dir /tmp/results_$NAME
  RC=$?
  [ $RC -ne 0 ] && { echo "ARM $NAME FAILED rc=$RC"; FAIL=1; }
  aws s3 sync /tmp/results_$NAME "$RES/$NAME/" --only-show-errors || FAIL=1
}

# ARMS format: "name:/channel/path[:base]" space-separated
for A in ${ARMS:?set ARMS}; do
  IFS=: read -r NAME DIR KIND <<< "$A"
  run_arm "$NAME" "$DIR" "$KIND"
done

[ $FAIL -ne 0 ] && { echo "one or more arms failed" | tee /opt/ml/output/failure; exit 1; }
echo "ALL ARMS DONE -> $RES"
