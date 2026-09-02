#!/usr/bin/env python3
"""Split a LeRobot v2.1 dataset into a train set and a held-out eval set.

Why this exists: open-loop evaluation on episodes the model trained on cannot
tell you anything about generalization. Hold out a few episodes here, run the
pipeline on the train output, prep + stage the holdout output separately, and
point evaluation/submit_eval_job.py --dataset-s3 at the holdout set.

Both outputs are RENUMBERED to contiguous episode indices starting at 0.
This is required, not cosmetic: the DreamZero GEAR converter iterates
`range(total_episodes)` and resolves files through info.json's path templates,
so a gap in episode numbering fails the prep stage. Renumbering rewrites, per
episode: the parquet filename, its `episode_index` column, its global `index`
column (kept globally consecutive in the new order — the trainer asserts
per-trajectory consecutiveness), and each camera's video filename. Frame
contents, timestamps, frame_index, task_index, and schema (incl. HF metadata)
are untouched. meta/split_manifest.json in each output records
new index -> source index for traceability.

meta/stats.json (aggregate feature stats) is copied as-is, NOT recomputed:
nothing in this pipeline consumes it — training statistics are regenerated
from the actual episodes by the prep stage's GEAR conversion.

Usage:
  python3 split_dataset.py --src /path/to/lerobot_v21 \
      --holdout 0,13,26,30,43,56,69,79 \
      --train-dst /path/to/out_train --holdout-dst /path/to/out_holdout
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # same filesystem: instant, no extra space
    except OSError:
        shutil.copy2(src, dst)


def rewrite_parquet(src_pq: Path, dst_pq: Path, new_ep_idx: int, index_start: int) -> int:
    """Copy one episode parquet with episode_index/index renumbered.

    Returns the number of rows (so the caller can advance the global index).
    """
    tbl = pq.read_table(src_pq)
    n = tbl.num_rows
    schema = tbl.schema
    for col, values in (("episode_index", [new_ep_idx] * n),
                        ("index", list(range(index_start, index_start + n)))):
        i = schema.get_field_index(col)
        if i < 0:
            if col == "index":
                continue  # some datasets omit the global index column
            sys.exit(f"ERROR: {src_pq} has no '{col}' column")
        tbl = tbl.set_column(i, schema.field(i),
                             pa.array(values, type=schema.field(i).type))
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, dst_pq)
    return n


def write_split(src: Path, dst: Path, episodes: list[dict], info: dict,
                label: str):
    """episodes: kept episodes.jsonl entries, in source-index order."""
    if dst.exists():
        sys.exit(f"ERROR: {dst} already exists — refusing to overwrite")
    (dst / "meta").mkdir(parents=True)

    chunk_size = info.get("chunks_size", 1000)
    data_tpl = info["data_path"]
    video_tpl = info["video_path"]
    vid_keys = [k for k in info["features"] if k.startswith("observation.images.")]

    manifest, index_start, total_frames = [], 0, 0
    new_entries = []
    for new_idx, entry in enumerate(episodes):
        old_idx = entry["episode_index"]
        old_fmt = {"episode_index": old_idx, "episode_chunk": old_idx // chunk_size}
        new_fmt = {"episode_index": new_idx, "episode_chunk": new_idx // chunk_size}

        n = rewrite_parquet(src / data_tpl.format(**old_fmt),
                            dst / data_tpl.format(**new_fmt),
                            new_idx, index_start)
        index_start += n
        total_frames += n
        if n != entry["length"]:
            sys.exit(f"ERROR: episode {old_idx}: parquet has {n} rows but "
                     f"episodes.jsonl says length {entry['length']}")

        for vk in vid_keys:
            src_mp4 = src / video_tpl.format(video_key=vk, **old_fmt)
            if not src_mp4.exists():
                sys.exit(f"ERROR: missing video {src_mp4}")
            link_or_copy(src_mp4, dst / video_tpl.format(video_key=vk, **new_fmt))

        new_entries.append({**entry, "episode_index": new_idx})
        manifest.append({"episode_index": new_idx, "source_episode_index": old_idx})

    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for e in new_entries:
            f.write(json.dumps(e) + "\n")
    with open(dst / "meta" / "split_manifest.json", "w") as f:
        json.dump({"split": label, "episodes": manifest}, f, indent=2)

    # episodes_stats.jsonl (per-episode stats, present in some v2.1 datasets)
    stats_jsonl = src / "meta" / "episodes_stats.jsonl"
    if stats_jsonl.exists():
        old_to_new = {m["source_episode_index"]: m["episode_index"] for m in manifest}
        with open(stats_jsonl) as fin, \
                open(dst / "meta" / "episodes_stats.jsonl", "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("episode_index") in old_to_new:
                    d["episode_index"] = old_to_new[d["episode_index"]]
                    fout.write(json.dumps(d) + "\n")

    for extra in ("tasks.jsonl", "stats.json"):
        if (src / "meta" / extra).exists():
            shutil.copy(src / "meta" / extra, dst / "meta" / extra)

    n_eps = len(new_entries)
    new_info = json.loads(json.dumps(info))
    new_info["total_episodes"] = n_eps
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = n_eps * len(vid_keys)
    new_info["total_chunks"] = (n_eps - 1) // chunk_size + 1 if n_eps else 0
    new_info["splits"] = {"train": f"0:{n_eps}"}
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    print(f"{label}: {n_eps} episodes, {total_frames} frames -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LeRobot v2.1 dataset dir")
    ap.add_argument("--holdout", required=True,
                    help="comma-separated source episode indices to hold out")
    ap.add_argument("--train-dst", required=True)
    ap.add_argument("--holdout-dst", required=True)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    info = json.load(open(src / "meta" / "info.json"))
    if info.get("codebase_version") != "v2.1":
        sys.exit(f"ERROR: dataset is {info.get('codebase_version')!r}, not v2.1")

    holdout = sorted({int(x) for x in args.holdout.split(",")})
    entries = []
    with open(src / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: e["episode_index"])
    have = {e["episode_index"] for e in entries}
    unknown = [i for i in holdout if i not in have]
    if unknown:
        sys.exit(f"ERROR: holdout episode(s) {unknown} not in the dataset")
    if len(holdout) >= len(entries):
        sys.exit("ERROR: holdout would leave no training episodes")

    train = [e for e in entries if e["episode_index"] not in holdout]
    held = [e for e in entries if e["episode_index"] in holdout]
    frac = sum(e["length"] for e in held) / sum(e["length"] for e in entries)
    print(f"holding out {len(held)}/{len(entries)} episodes "
          f"({frac:.1%} of frames): {holdout}")

    write_split(src, Path(args.train_dst).resolve(), train, info, "train")
    write_split(src, Path(args.holdout_dst).resolve(), held, info, "holdout")
    print("SPLIT COMPLETE")


if __name__ == "__main__":
    main()
