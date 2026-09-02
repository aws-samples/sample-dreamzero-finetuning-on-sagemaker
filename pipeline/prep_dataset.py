#!/usr/bin/env python3
"""Config-driven LeRobot v2.1 -> GEAR/yam dataset prep.

Generalizes data_prep/prepare_aloha_yam.sh: everything robot-specific comes
from an embodiment config (configs/*.yaml). Golden test: with
configs/aloha_bimanual_14dim.yaml this reproduces the validated aloha_yam
dataset's meta/ byte-for-byte (modulo JSON key order — compared parsed).

Steps:
  1. copy parquet + tasks/episodes meta
  2. re-encode videos to h264/yuv420p iff codec matches config.reencode_codecs,
     renaming cameras per config.camera_map (unlisted cameras dropped)
  3. rewrite meta/info.json (renamed keys, dropped cameras, codec metadata)
  4. run dreamzero's convert_lerobot_to_gear.py (embodiment tag, dim slices)
  5. patch modality.json annotation to {"task": {"original_key": <task_key>}}
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import os
# path to the dreamzero repo clone (has scripts/data/convert_lerobot_to_gear.py)
DREAMZERO = Path(os.environ.get("DREAMZERO_REPO", "./dreamzero")).resolve()


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_meta(feature):
    """The nested video-metadata dict of a camera feature, whichever key holds it.

    The key varies by dataset vintage: hub datasets converted from LeRobot v1.x
    (e.g. lerobot/aloha_* at rev v2.1) use "video_info", datasets recorded by
    newer lerobot versions (and v3-style trees) use "info". An av1 dataset in
    the newer format would otherwise silently skip re-encoding — checking one
    key only is exactly the kind of no-error breakage this pipeline exists to
    prevent (dreamzero's own reader tries both keys for the same reason).
    Returns a throwaway {} when neither is present, so reads see no codec and
    writes are no-ops.
    """
    for key in ("video_info", "info"):
        if isinstance(feature.get(key), dict):
            return feature[key]
    return {}


def detect_codec(info, feature_key):
    return video_meta(info["features"][feature_key]).get("video.codec", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LeRobot v2.1 dataset dir")
    ap.add_argument("--dst", required=True, help="output GEAR dataset dir")
    ap.add_argument("--config", required=True, help="embodiment config yaml")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    if not (DREAMZERO / "scripts" / "data" / "convert_lerobot_to_gear.py").exists():
        sys.exit(f"ERROR: DreamZero repo not found at {DREAMZERO} — "
                 "git clone https://github.com/dreamzero0/dreamzero.git and "
                 "set DREAMZERO_REPO to the clone path")

    # resolve() both: the GEAR conversion below runs with cwd=DREAMZERO, so a
    # relative --dst would silently point the converter at the wrong tree
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    cfg = yaml.safe_load(open(args.config))
    cam_map = cfg["camera_map"]

    info = json.load(open(src / "meta" / "info.json"))
    if info.get("codebase_version") != "v2.1":
        sys.exit(f"ERROR: dataset is {info.get('codebase_version')!r}, not v2.1 — "
                 "run the v3->v2.1 converter first (pipeline stage 1)")

    # validate config against the dataset before touching anything
    feats = info["features"]
    for cam in cam_map:
        key = f"observation.images.{cam}"
        if key not in feats:
            sys.exit(f"ERROR: config camera '{cam}' not in dataset features "
                     f"(have: {[k for k in feats if k.startswith('observation.images.')]})")
    for group, feat in (("state_keys", "observation.state"),
                        ("action_keys", "action")):
        dim = feats[feat]["shape"][0]
        max_end = max(e for _, e in cfg[group].values())
        if max_end > dim:
            sys.exit(f"ERROR: {group} slice end {max_end} exceeds "
                     f"{feat} dim {dim}")

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)
    (dst / "meta").mkdir()
    for chunk in sorted((src / "data").glob("chunk-*")):
        shutil.copytree(chunk, dst / "data" / chunk.name)
    for f in ("tasks.jsonl", "episodes.jsonl"):
        shutil.copy(src / "meta" / f, dst / "meta" / f)

    # --- videos (all chunks: >1000 episodes spill into chunk-001+) ---
    ff = ffmpeg_exe()
    jobs = []
    video_chunks = sorted((src / "videos").glob("chunk-*"))
    if not video_chunks:
        sys.exit(f"ERROR: no videos/chunk-* dirs in {src}")
    for cam, view in cam_map.items():
        src_key = f"observation.images.{cam}"
        codec = detect_codec(info, src_key)
        reenc = codec in cfg.get("reencode_codecs", [])
        total = 0
        for chunk in video_chunks:
            out_dir = dst / "videos" / chunk.name / f"observation.images.{view}"
            out_dir.mkdir(parents=True)
            for mp4 in sorted((chunk / src_key).glob("*.mp4")):
                total += 1
                if reenc:
                    jobs.append([ff, "-y", "-loglevel", "error", "-i", str(mp4),
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-crf", "23", "-g", "10", str(out_dir / mp4.name)])
                else:
                    jobs.append(["cp", str(mp4), str(out_dir / mp4.name)])
        print(f"{cam} -> {view}: codec={codec or '?'} reencode={reenc} "
              f"({total} files across {len(video_chunks)} chunk(s))")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(lambda c: subprocess.run(c, check=True), jobs):
            pass
    n = len(list((dst / "videos").rglob("*.mp4")))
    expected = info["total_episodes"] * len(cam_map)
    if n != expected:  # not assert: must survive python -O
        sys.exit(f"ERROR: video count {n} != expected {expected}")

    # --- info.json rewrite ---
    new_feats = {}
    for k, v in feats.items():
        if k.startswith("observation.images."):
            cam = k.replace("observation.images.", "")
            if cam not in cam_map:
                continue  # dropped camera
            v = json.loads(json.dumps(v))
            if detect_codec(info, k) in cfg.get("reencode_codecs", []):
                video_meta(v)["video.codec"] = "h264"
            new_feats[f"observation.images.{cam_map[cam]}"] = v
        else:
            new_feats[k] = v
    info["features"] = new_feats
    info["total_videos"] = expected
    json.dump(info, open(dst / "meta" / "info.json", "w"), indent=4)

    # --- GEAR conversion (no --force: preserve copied tasks/episodes jsonl) ---
    subprocess.run(
        [sys.executable, "scripts/data/convert_lerobot_to_gear.py",
         "--dataset-path", str(dst),
         "--embodiment-tag", cfg["embodiment_tag"],
         "--state-keys", json.dumps(cfg["state_keys"], separators=(",", ":")),
         "--action-keys", json.dumps(cfg["action_keys"], separators=(",", ":")),
         "--relative-action-keys", *cfg["relative_action_keys"],
         "--task-key", cfg["task_key"],
         "--action-horizon", str(cfg["action_horizon"])],
        cwd=DREAMZERO, check=True)

    # --- annotation patch ---
    mod_path = dst / "meta" / "modality.json"
    mod = json.load(open(mod_path))
    mod["annotation"] = {"task": {"original_key": cfg["task_key"]}}
    json.dump(mod, open(mod_path, "w"), indent=4)

    print(f"PREP COMPLETE: {dst} (embodiment={cfg['embodiment_tag']}, fps={cfg['fps']})")


if __name__ == "__main__":
    main()
