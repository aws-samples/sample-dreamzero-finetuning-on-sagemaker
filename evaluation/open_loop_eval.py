#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA Corporation. All rights reserved.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is derived from scripts/open_loop_yam.py in NVIDIA DreamZero
# (https://github.com/dreamzero0/dreamzero, commit ab790c1), which is
# licensed under the Apache License 2.0. Modifications: the robot layout
# (state/action slices, camera map) is read from the dataset's
# meta/modality.json instead of being hardcoded; deterministic evenly-spaced
# frame sampling; stock-LeRobot task_index resolution; per-key/per-dim MSE
# written to mse.txt. See THIRD-PARTY-LICENSES at the repo root.
"""Offline open-loop evaluation for DreamZero on a GEAR/yam-prepped dataset.

Loads a model checkpoint directly (no server needed), reads the dataset
(parquet + MP4), runs inference, and compares predicted vs ground-truth actions.

Usage:
    python scripts/open_loop_eval.py \
        --model_path /path/to/checkpoint \
        --dataset_path /path/to/gear_dataset \
        --device cuda:0 \
        --num_samples 200
"""

import torch._dynamo
torch._dynamo.config.disable = True

import argparse
import glob
import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


# ---------------------------------------------------------------------------
# Robot layout — read from the dataset's meta/modality.json, the same file the
# prep stage writes and training consumes, so eval slices cannot disagree with
# the embodiment config used at training time. (The original script hardcoded
# a 16-dim bimanual layout here; on a robot with any other layout that
# silently mis-sliced ground truth.)
# ---------------------------------------------------------------------------

VIDEO_CAMERAS = {}   # "video.<name>" -> "observation.images.<name>" folder
STATE_SLICES = {}    # "state.<name>" -> (start, end) into the packed vector
ACTION_SLICES = {}   # "action.<name>" -> (start, end)
ACTION_KEY_ORDER = []


def load_layout(dataset_path: str):
    """Populate slice/camera globals from <dataset>/meta/modality.json."""
    mod_path = os.path.join(dataset_path, "meta", "modality.json")
    with open(mod_path) as f:
        mod = json.load(f)
    # rebuild, don't accumulate: a second call (another dataset in the same
    # process) must not merge stale slices or duplicate ACTION_KEY_ORDER
    VIDEO_CAMERAS.clear()
    STATE_SLICES.clear()
    ACTION_SLICES.clear()
    ACTION_KEY_ORDER.clear()
    for name, spec in mod["state"].items():
        STATE_SLICES[f"state.{name}"] = (int(spec["start"]), int(spec["end"]))
    for name, spec in mod["action"].items():
        ACTION_SLICES[f"action.{name}"] = (int(spec["start"]), int(spec["end"]))
    ACTION_KEY_ORDER.extend(f"action.{n}" for n in mod["action"])
    for name, spec in mod["video"].items():
        VIDEO_CAMERAS[f"video.{name}"] = spec["original_key"]
    if not (STATE_SLICES and ACTION_SLICES and VIDEO_CAMERAS):
        raise ValueError(f"{mod_path} is missing state/action/video entries — "
                         "was the dataset prepped with prep_dataset.py?")
    print(f"Layout from {mod_path}:")
    print(f"  state:  {STATE_SLICES}")
    print(f"  action: {ACTION_SLICES}")
    print(f"  video:  {list(VIDEO_CAMERAS)}")


# ---------------------------------------------------------------------------
# Dataset reader  (LeRobot chunked format)
# ---------------------------------------------------------------------------

class YAMDataset:
    """Reads LeRobot-style chunked parquet + MP4."""

    def __init__(self, dataset_path: str):
        self.root = dataset_path

        data_dir = os.path.join(dataset_path, "data")
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "**", "episode_*.parquet"), recursive=True))
        if not parquet_files:
            raise FileNotFoundError(f"No episode_*.parquet found under {data_dir}")

        self.episodes = []
        self.episode_names = []  # parquet stem, e.g. "episode_000012"
        self.cum_lengths = [0]
        for pf in parquet_files:
            table = pq.read_table(pf)
            self.episodes.append(table)
            self.episode_names.append(os.path.splitext(os.path.basename(pf))[0])
            self.cum_lengths.append(self.cum_lengths[-1] + table.num_rows)
        self.total_rows = self.cum_lengths[-1]

        # Index every camera's mp4s by episode name across ALL chunk dirs —
        # positional indexing would silently pair the wrong video with a row
        # when episode numbering has gaps or spills past chunk-000.
        videos_root = os.path.join(dataset_path, "videos")
        self.video_files = {}  # server_key -> {episode_name: mp4 path}
        for server_key, folder_name in VIDEO_CAMERAS.items():
            found = {}
            for d in sorted(glob.glob(os.path.join(videos_root, "**", folder_name),
                                      recursive=True)):
                for mp4 in glob.glob(os.path.join(d, "*.mp4")):
                    found[os.path.splitext(os.path.basename(mp4))[0]] = mp4
            if not found:
                raise FileNotFoundError(
                    f"camera folder '{folder_name}' (for {server_key}) has no "
                    f"videos under {videos_root} — an observation missing a "
                    "view the model was trained on would silently degrade "
                    "every prediction")
            self.video_files[server_key] = found

        print(f"YAMDataset: {len(self.episodes)} episodes, "
              f"{self.total_rows} rows, {len(self.video_files)} cameras")

    def __len__(self):
        return self.total_rows

    def _locate(self, idx):
        for ep in range(len(self.episodes)):
            if idx < self.cum_lengths[ep + 1]:
                return ep, idx - self.cum_lengths[ep]
        raise IndexError(f"Index {idx} out of range ({self.total_rows})")

    def get_state(self, idx) -> np.ndarray:
        ep, row = self._locate(idx)
        return np.array(self.episodes[ep].column("observation.state")[row].as_py(), dtype=np.float64)

    def get_action(self, idx) -> np.ndarray:
        ep, row = self._locate(idx)
        return np.array(self.episodes[ep].column("action")[row].as_py(), dtype=np.float64)

    def get_task(self, idx) -> str:
        ep, row = self._locate(idx)
        try:
            return str(self.episodes[ep].column("annotation.task")[row].as_py())
        except Exception:
            pass
        # stock LeRobot: integer task_index resolved through meta/tasks.jsonl
        try:
            ti = int(self.episodes[ep].column("task_index")[row].as_py())
            if not hasattr(self, "_tasks"):
                import json as _json
                self._tasks = {}
                with open(os.path.join(self.root, "meta", "tasks.jsonl")) as f:
                    for line in f:
                        d = _json.loads(line)
                        self._tasks[int(d["task_index"])] = d["task"]
            return self._tasks.get(ti, "")
        except Exception:
            return ""

    def get_frame(self, idx, server_key) -> np.ndarray:
        """Read one video frame → (H, W, 3) uint8 RGB."""
        ep, row = self._locate(idx)
        name = self.episode_names[ep]
        mp4 = self.video_files[server_key].get(name)
        if mp4 is None:
            raise FileNotFoundError(
                f"no video for {name} in camera {server_key} — parquet and "
                "video trees disagree")
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, row)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {row} from {mp4}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_obs(dataset: YAMDataset, idx: int, prompt: str) -> dict:
    """Build an obs dict matching what GrootSimPolicy.forward() expects."""
    obs = {}

    for server_key in dataset.video_files:
        frame = dataset.get_frame(idx, server_key)
        obs[server_key] = frame[np.newaxis, ...].astype(np.uint8)  # (1, H, W, C)

    state = dataset.get_state(idx)
    for key, (start, end) in STATE_SLICES.items():
        obs[key] = state[start:end].reshape(1, -1).astype(np.float64)  # (1, D)

    obs["annotation.task"] = prompt

    return obs


def get_gt_action_dict(dataset: YAMDataset, idx: int) -> dict:
    """Split the flat GT action vector into per-key arrays."""
    action_flat = dataset.get_action(idx)
    gt = {}
    for key in ACTION_KEY_ORDER:
        s, e = ACTION_SLICES[key]
        gt[key] = action_flat[s:e]
    return gt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(all_preds, all_gts, key_names, output_dir):
    """Plot pred vs gt for each action dimension across all keys."""
    pred_flat = np.concatenate([all_preds[k] for k in key_names], axis=-1)
    gt_flat = np.concatenate([all_gts[k] for k in key_names], axis=-1)
    D = pred_flat.shape[1]
    mse_dim = np.mean((pred_flat - gt_flat) ** 2, axis=0)

    for d in range(D):
        plt.figure(figsize=(10, 4))
        plt.plot(gt_flat[:, d], label="gt", alpha=0.8)
        plt.plot(pred_flat[:, d], label="pred", alpha=0.8)
        plt.title(f"Action dim {d}  (MSE={mse_dim[d]:.6f})")
        plt.xlabel("sample index"); plt.ylabel("value")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"action_dim_{d}.png"), dpi=150)
        plt.close()

    ncols = 4
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    overall_mse = float(np.mean(mse_dim))
    fig.suptitle(f"All action dims  (overall MSE={overall_mse:.6f})", fontsize=14)
    for d in range(D):
        ax = axes[d // ncols][d % ncols]
        ax.plot(gt_flat[:, d], label="gt", alpha=0.7, lw=0.8)
        ax.plot(pred_flat[:, d], label="pred", alpha=0.7, lw=0.8)
        ax.set_title(f"dim {d} (MSE={mse_dim[d]:.4f})", fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
        if d == 0: ax.legend(fontsize=7)
    for d in range(D, nrows * ncols):
        axes[d // ncols][d % ncols].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "all_action_dims.png"), dpi=200)
    plt.close(fig)

    # Per-key summary plot
    fig2, axes2 = plt.subplots(1, len(key_names), figsize=(5 * len(key_names), 4), squeeze=False)
    for i, k in enumerate(key_names):
        ax = axes2[0][i]
        p, g = all_preds[k], all_gts[k]
        for d in range(p.shape[1]):
            ax.plot(g[:, d], '--', alpha=0.5, lw=0.8)
            ax.plot(p[:, d], alpha=0.7, lw=0.8)
        key_mse = float(np.mean((p - g) ** 2))
        ax.set_title(f"{k}\nMSE={key_mse:.6f}", fontsize=9)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
    fig2.suptitle("Per-key pred (solid) vs gt (dashed)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(os.path.join(output_dir, "per_key_summary.png"), dpi=200)
    plt.close(fig2)

    return mse_dim, overall_mse


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    load_layout(args.dataset_path)

    # Single-process distributed init (GrootSimPolicy uses dist.get_rank())
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    print(f"Loading model from {args.model_path} ...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
    )
    print("Model loaded.")

    dataset = YAMDataset(args.dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)

    num = min(args.num_samples, len(dataset))
    # deterministic, evenly-spaced frame set spanning ALL episodes — identical
    # across eval arms by construction (no RNG)
    sample_indices = [int(i * len(dataset) / num) for i in range(num)]
    preds_per_key = {k: [] for k in ACTION_KEY_ORDER}
    gts_per_key = {k: [] for k in ACTION_KEY_ORDER}
    times = []

    print(f"\nEvaluating {num} samples (start={args.start_idx}) ...")
    print("-" * 60)

    for i in range(num):
        idx = sample_indices[i] if args.spread_samples else (args.start_idx + i)

        prompt = args.prompt
        if args.use_dataset_prompt:
            task = dataset.get_task(idx)
            if task:
                prompt = task

        obs = build_obs(dataset, idx, prompt)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        gt = get_gt_action_dict(dataset, idx)

        for k in ACTION_KEY_ORDER:
            if k in result.act:
                pred_val = result.act[k]
                if isinstance(pred_val, torch.Tensor):
                    pred_val = pred_val.cpu().numpy()
                # First timestep: squeeze may collapse (1,24,1) -> (24,)
                pred_val = np.atleast_1d(pred_val[0]).flatten()
                if pred_val.shape != gt[k].shape:
                    raise ValueError(
                        f"{k}: prediction width {pred_val.shape} != ground-truth "
                        f"width {gt[k].shape} — the checkpoint's embodiment does "
                        "not match this dataset's meta/modality.json layout; an "
                        "MSE computed across mismatched dims would be meaningless")
                preds_per_key[k].append(pred_val)
                gts_per_key[k].append(gt[k])

        if i % args.log_every == 0:
            if i == 0:
                print(f"  Action keys in output: {list(result.act.keys())}")
                for k in ACTION_KEY_ORDER:
                    if k in result.act:
                        v = result.act[k]
                        shape = v.shape if hasattr(v, 'shape') else "?"
                        print(f"    {k}: pred_shape={shape}, gt_shape={gt[k].shape}")
            print(f"  [{i:>5d}/{num}] idx={idx} infer={elapsed:.3f}s prompt={prompt!r:.60}")

    # Stack results
    valid_keys = [k for k in ACTION_KEY_ORDER if len(preds_per_key[k]) > 0]
    if not valid_keys:
        raise RuntimeError(
            "no predictions: none of the dataset's action keys "
            f"{ACTION_KEY_ORDER} appear in the model output "
            f"{list(result.act.keys())} — checkpoint/embodiment mismatch "
            "(a base arm needs --config-donor / BASE_CONFIG_DONOR). "
            "Exiting nonzero so the job does not report success with no "
            "mse.txt in S3.")

    stacked_preds = {k: np.stack(preds_per_key[k]) for k in valid_keys}
    stacked_gts = {k: np.stack(gts_per_key[k]) for k in valid_keys}

    pred_all = np.concatenate([stacked_preds[k] for k in valid_keys], axis=-1)
    gt_all = np.concatenate([stacked_gts[k] for k in valid_keys], axis=-1)
    overall_mse = float(np.mean((pred_all - gt_all) ** 2))

    print(f"\n{'='*60}")
    print(f"Overall MSE: {overall_mse:.6f}  |  Avg inference time: {np.mean(times):.4f}s")
    for k in valid_keys:
        k_mse = float(np.mean((stacked_preds[k] - stacked_gts[k]) ** 2))
        print(f"  {k}: MSE={k_mse:.6f}")
    print(f"{'='*60}")

    mse_dim, _ = save_plots(stacked_preds, stacked_gts, valid_keys, args.output_dir)

    with open(os.path.join(args.output_dir, "mse.txt"), "w") as f:
        f.write(f"overall_mse,{overall_mse}\n")
        for k in valid_keys:
            k_mse = float(np.mean((stacked_preds[k] - stacked_gts[k]) ** 2))
            f.write(f"{k},{k_mse}\n")
        for d, v in enumerate(mse_dim):
            f.write(f"dim_{d},{v}\n")

    print(f"Results saved to {os.path.abspath(args.output_dir)}/")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", required=True,
                   help="Path to model checkpoint dir (contains config.json, model.safetensors, experiment_cfg/)")
    p.add_argument("--dataset_path", required=True,
                   help="Root of YAM dataset (contains data/, videos/, meta/)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--embodiment_tag", default="yam",
                   help="EmbodimentTag value the checkpoint was trained with "
                        "(yam for the GEAR bimanual recipe, oxe_droid for DROID)")
    p.add_argument("--prompt", default="pick up the object")
    p.add_argument("--use_dataset_prompt", action="store_true",
                   help="Read task annotation from parquet instead of --prompt")
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--spread_samples", action="store_true",
                   help="evenly-spaced deterministic frames across all episodes")
    p.add_argument("--output_dir", default="results_yam")
    p.add_argument("--log_every", type=int, default=10)
    main_args = p.parse_args()
    evaluate(main_args)


if __name__ == "__main__":
    main()
