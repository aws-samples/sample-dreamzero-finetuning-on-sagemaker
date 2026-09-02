#!/usr/bin/env python3
"""Run one observation -> action-chunk prediction from a MERGED checkpoint.

This is the minimal "how do I actually use the weights" path: load the merged
model the same way evaluation does (inheriting its checkpoint-portability
fixes is up to the caller — run inside the training container with the wan/
tokenizer dirs present, exactly like evaluation/run_eval_in_job.sh sets up),
build an observation from a dataset frame, and print the predicted action
chunk (action_horizon x action_dim) as JSON.

Why not a SageMaker real-time endpoint: the merged checkpoint is ~92GB fp32,
needs 80GB+ VRAM, and produces multi-second action chunks for closed-loop
control — a persistent GPU host that holds the model resident fits that shape;
request/response hosting does not.

Usage (inside the training container, or any env where the DreamZero repo and
its deps import):
    python inference/predict.py \
        --model_path /path/to/merged_checkpoint \
        --dataset_path /path/to/gear_dataset \
        --index 0 \
        --output actions.json

The observation is taken from the dataset frame at --index; the robot layout
(state/action slices, cameras) comes from the dataset's meta/modality.json.
NEVER point --model_path at a raw LoRA directory — merge first (see the main
README: the un-merged path silently composes onto the wrong base and is worse
than not fine-tuning at all).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Reuse the evaluation module's dataset reader, obs builder, and
# modality.json-driven layout so inference and evaluation can never disagree
# about how an observation is constructed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from open_loop_eval import (  # noqa: E402
    YAMDataset, build_obs, load_layout, ACTION_KEY_ORDER,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="MERGED checkpoint dir (config.json + safetensors + "
                         "experiment_cfg/) — never a raw LoRA dir")
    ap.add_argument("--dataset_path", required=True,
                    help="GEAR/yam-prepped dataset (source of the observation)")
    ap.add_argument("--index", type=int, default=0,
                    help="dataset frame index to build the observation from")
    ap.add_argument("--prompt", default=None,
                    help="task instruction (default: the frame's own annotation)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embodiment_tag", default="yam",
                    help="EmbodimentTag the checkpoint was trained with "
                         "(yam for GEAR bimanual, oxe_droid for DROID)")
    ap.add_argument("--output", default=None,
                    help="write the action chunk to this JSON file "
                         "(default: stdout only)")
    args = ap.parse_args()

    # A raw (un-merged) LoRA checkpoint is identified by its experiment_cfg:
    # the trainer writes save_lora_only: true / train_architecture: lora, and
    # merge_lora.py rewrites both on merge. Loading a raw dir silently routes
    # through the repo's load_lora path, which composes onto the wrong base —
    # measured 9.9x worse than the merged checkpoint.
    conf_path = Path(args.model_path) / "experiment_cfg" / "conf.yaml"
    if conf_path.exists():
        conf = conf_path.read_text()
        if ("save_lora_only: true" in conf
                or "train_architecture: lora" in conf):
            sys.exit("ABORT: this is a raw LoRA checkpoint (experiment_cfg/"
                     "conf.yaml says save_lora_only/train_architecture: lora)."
                     " Merge it first — run_pipeline.py's merge stage, or "
                     "pipeline/merge_lora.py. Serving a raw LoRA dir silently "
                     "produces a badly degraded model.")

    load_layout(args.dataset_path)

    import torch
    import torch.distributed as dist
    from tianshou.data import Batch
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    print(f"Loading model from {args.model_path} ...")
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag(args.embodiment_tag),
                            model_path=args.model_path, device=args.device)

    dataset = YAMDataset(args.dataset_path)
    prompt = args.prompt or dataset.get_task(args.index) or "pick up the object"
    obs = build_obs(dataset, args.index, prompt)

    with torch.inference_mode():
        result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))

    import numpy as np
    chunk = {}
    for k in ACTION_KEY_ORDER:
        if k in result.act:
            v = result.act[k]
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            # normalize to (action_horizon, dim) — the FULL chunk. The policy
            # squeezes the batch dim, so (horizon, dim) arrives directly and a
            # 1-dim key (gripper) collapses to (horizon,); a 3-d array still
            # carries the batch dim.
            arr = np.asarray(v)
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim == 1:
                arr = arr[:, np.newaxis]
            chunk[k] = arr.tolist()
    if not chunk:
        sys.exit(f"ABORT: model output {list(result.act.keys())} contains "
                 f"none of the dataset's action keys {ACTION_KEY_ORDER} — "
                 "checkpoint/embodiment mismatch")

    out = {"prompt": prompt, "frame_index": args.index, "actions": chunk}
    print(json.dumps(out, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
