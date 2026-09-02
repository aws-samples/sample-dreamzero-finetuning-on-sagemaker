#!/usr/bin/env python3
"""Validate a LeRobot v2.1 dataset against an embodiment config BEFORE any
compute is spent. Every check here catches a mistake that otherwise trains a
broken model without raising an error:

  1. state_keys/action_keys slice bounds vs the dataset's actual vector dims
  2. camera_map entries exist; exactly 3 views (the trainer runs num_views=3)
  3. config fps == dataset fps — a mismatch means the config was copied from
     a different robot. (fps itself is inert at train time under the trainer's
     decord backend — frame/action alignment comes from the parquet timestamps
     matching video PTS — but a stale config rarely stops at fps)
  4. task annotation column resolvable (task_key or task_index + tasks.jsonl)
  5. action-equals-state degeneracy scan: if the action labels are just a copy
     of the state, training "succeeds" and open-loop MSE looks excellent —
     because a model that echoes its input scores perfectly — but the policy
     has learned nothing. This is a data-collection bug worth failing loudly on.
  6. completeness: every episode listed in meta/episodes.jsonl has its parquet
     and all its per-camera videos on disk. The trainer builds its trajectory
     list from episodes.jsonl, not from the files present, so a partial
     download passes every check above and then dies mid-job with a bare
     FileNotFoundError — after the queue wait and the channel download.

Usage:
  python3 validate_dataset.py --src /path/to/lerobot_v21 --config configs/x.yaml \
      [--fail-on-degeneracy] [--sample-rows 2000]

Exit 0 = safe to proceed. Exit 1 = a check failed (message says which).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def err(msg):
    print(f"VALIDATION FAIL: {msg}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LeRobot v2.1 dataset dir")
    ap.add_argument("--config", required=True, help="embodiment config yaml")
    ap.add_argument("--sample-rows", type=int, default=2000,
                    help="rows sampled for the action-vs-state degeneracy scan")
    ap.add_argument("--fail-on-degeneracy", action="store_true",
                    help="treat action==state labels as an error, not a warning")
    ap.add_argument("--gear", action="store_true",
                    help="dataset is pre-prepped GEAR format: take the "
                         "state/action layout and cameras from its own "
                         "meta/modality.json instead of the config yaml")
    args = ap.parse_args()

    src = Path(args.src)
    cfg = yaml.safe_load(open(args.config))
    info = json.load(open(src / "meta" / "info.json"))
    feats = info["features"]
    rc = 0

    if args.gear:
        # the dataset is its own source of truth for slices and cameras — the
        # yaml only pins fps and the training recipe. Derive the layout so the
        # same bound and degeneracy checks below still run.
        mod = json.load(open(src / "meta" / "modality.json"))
        cfg["state_keys"] = {n: [s["start"], s["end"]]
                             for n, s in mod["state"].items()}
        cfg["action_keys"] = {n: [s["start"], s["end"]]
                              for n, s in mod["action"].items()}
        cfg["camera_map"] = {v["original_key"].replace("observation.images.", ""): n
                             for n, v in mod["video"].items()}
        cfg.setdefault("task_key", "task_index")

    # --- 1. slice bounds ---
    state_dim = feats["observation.state"]["shape"][0]
    action_dim = feats["action"]["shape"][0]
    for group, dim in (("state_keys", state_dim), ("action_keys", action_dim)):
        ends = {name: se[1] for name, se in cfg[group].items()}
        over = {n: e for n, e in ends.items() if e > dim}
        if over:
            rc |= err(f"{group} slice(s) {over} exceed the dataset's "
                      f"{group.split('_')[0]} dim {dim} — these would silently "
                      "read wrong/empty values")
        covered = max(ends.values())
        if covered < dim:
            print(f"  note: {group} cover dims [0,{covered}) of {dim} — "
                  f"dims [{covered},{dim}) will be dropped")

    # --- 2. cameras ---
    have = [k.replace("observation.images.", "") for k in feats
            if k.startswith("observation.images.")]
    missing = [c for c in cfg["camera_map"] if c not in have]
    if missing:
        rc |= err(f"camera_map source(s) {missing} not in dataset (have: {have})")
    if len(cfg["camera_map"]) != 3:
        rc |= err(f"camera_map has {len(cfg['camera_map'])} entries; the trainer "
                  "runs with num_views=3, so exactly 3 views must be mapped "
                  "(top = scene, left/right = wrist)")

    # --- 3. fps ---
    ds_fps = info.get("fps")
    if ds_fps is not None and float(cfg["fps"]) != float(ds_fps):
        rc |= err(f"config fps={cfg['fps']} but the dataset records at "
                  f"fps={ds_fps} — was this config copied from a different "
                  "robot? (fps itself is inert under the trainer's decord "
                  "backend, but the slices and cameras next to it rarely "
                  "survive a copy)")

    # --- 4. task annotation ---
    task_key = cfg["task_key"]
    if task_key == "task_index":
        if not (src / "meta" / "tasks.jsonl").exists():
            rc |= err("task_key=task_index but meta/tasks.jsonl is missing")
    elif task_key not in feats:
        rc |= err(f"task_key '{task_key}' is not a dataset feature "
                  f"(stock LeRobot uses task_index + meta/tasks.jsonl)")

    # --- 5. action-vs-state degeneracy scan ---
    import pyarrow.parquet as pq
    # only compare slices of equal width; a shared key with different state
    # and action widths can't be an echo and would break broadcasting
    shared = [k for k in cfg["action_keys"] if k in cfg["state_keys"]
              and (cfg["state_keys"][k][1] - cfg["state_keys"][k][0])
              == (cfg["action_keys"][k][1] - cfg["action_keys"][k][0])]
    states, actions = [], []
    remaining = args.sample_rows
    for pf in sorted((src / "data").rglob("episode_*.parquet")):
        if remaining <= 0:
            break
        t = pq.read_table(pf, columns=["observation.state", "action"])
        n = min(t.num_rows, remaining)
        if n == 0:
            continue
        states.append(np.array(t.column("observation.state").to_pylist()[:n]))
        actions.append(np.array(t.column("action").to_pylist()[:n]))
        remaining -= n
    if not states:
        # a missing scan must not read as a pass — this is the check that
        # catches echo-labeled data
        rc |= err("degeneracy scan found no data/**/episode_*.parquet rows — "
                  "is this really a LeRobot v2.1 dataset? (v3 layouts must be "
                  "converted first; non-standard filenames aren't supported)")
    elif shared:
        state = np.concatenate(states)
        action = np.concatenate(actions)
        eq_rows = None
        for key in shared:
            ss, se = cfg["state_keys"][key]
            as_, ae = cfg["action_keys"][key]
            close = np.all(np.isclose(state[:, ss:se], action[:, as_:ae],
                                      atol=1e-6), axis=1)
            eq_rows = close if eq_rows is None else (eq_rows & close)
        frac = float(np.mean(eq_rows))
        print(f"  degeneracy scan: action == state on {frac:.1%} of "
              f"{len(state)} sampled rows")
        if frac > 0.95:
            msg = (f"action labels equal the state on {frac:.1%} of sampled rows. "
                   "A model trained on this only learns to echo its input; "
                   "open-loop MSE will look excellent while the policy does "
                   "nothing. Fix the data collection (actions should be the "
                   "COMMANDED targets, not the measured state).")
            if args.fail_on_degeneracy:
                rc |= err(msg)
            else:
                print(f"  WARNING: {msg}")
                print("  (pass --fail-on-degeneracy to make this an error)")

    # --- 6. completeness: episodes.jsonl vs files on disk ---
    # The trainer enumerates trajectories from episodes.jsonl and resolves each
    # one through info.json's own path templates. Anything listed there but
    # absent on disk is a FileNotFoundError once training reaches that shard —
    # so resolve the same way and report every gap now.
    ep_file = src / "meta" / "episodes.jsonl"
    if not ep_file.exists():
        rc |= err("meta/episodes.jsonl is missing — the trainer builds its "
                  "trajectory list from it")
    else:
        data_tpl = info.get("data_path")
        video_tpl = info.get("video_path")
        chunk_size = info.get("chunks_size", 1000)
        vid_keys = [k for k in feats if k.startswith("observation.images.")]
        missing_data, missing_video, n_eps = [], [], 0
        with open(ep_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx = json.loads(line).get("episode_index")
                if idx is None:
                    continue
                n_eps += 1
                fmt = {"episode_index": idx, "episode_chunk": idx // chunk_size}
                if data_tpl and not (src / data_tpl.format(**fmt)).exists():
                    missing_data.append(idx)
                for vk in vid_keys if video_tpl else []:
                    if not (src / video_tpl.format(video_key=vk, **fmt)).exists():
                        missing_video.append((idx, vk))

        def _sample(xs, n=5):
            return ", ".join(str(x) for x in xs[:n]) + (" ..." if len(xs) > n else "")

        if missing_data:
            rc |= err(f"{len(missing_data)} of {n_eps} episodes listed in "
                      f"meta/episodes.jsonl have no parquet on disk "
                      f"(e.g. {_sample(missing_data)}). The trainer reads that "
                      "list, not the directory, so this fails mid-job. Either "
                      "finish the download or trim episodes.jsonl (and "
                      "info.json's totals) to the episodes you actually have.")
        if missing_video:
            rc |= err(f"{len(missing_video)} episode/camera videos listed in "
                      f"meta/episodes.jsonl are absent "
                      f"(e.g. {_sample(missing_video)})")
        if not missing_data and not missing_video:
            print(f"  completeness: all {n_eps} episodes in episodes.jsonl "
                  f"have their parquet + {len(vid_keys)} videos on disk")

    if rc:
        sys.exit(1)
    print(f"VALIDATION OK: {src} matches {args.config}")


if __name__ == "__main__":
    main()
