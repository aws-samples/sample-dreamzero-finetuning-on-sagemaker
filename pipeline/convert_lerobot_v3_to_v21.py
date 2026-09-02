#!/usr/bin/env python3
"""
Convert a LeRobot **v3.0** dataset (chunked/aggregated layout) to the **v2.1**
layout this pipeline consumes, entirely locally — no AWS, no GPU.

v3 aggregates many episodes into shared files, and *how* it aggregates varies
per dataset, so treat this as a WORKED EXAMPLE rather than a universal tool. It
is verified on bimanual, 3-camera OpenArms v3 trees (16-dim state/action) in
two different aggregation styles — one row group per episode, and all episodes
in a single row group — and adapts to either. For a materially different layout
(different camera count, single arm, per-episode video files) adapt the split
and cut logic below.

What it does:
  1. Splits the aggregated data parquet(s) into per-episode parquets
     (data/chunk-000/episode_XXXXXX.parquet) by grouping on the
     `episode_index` column, cross-checked against `meta/episodes`'
     `dataset_from_index`/`dataset_to_index`. Row groups are NOT used as the
     episode boundary: some writers emit one row group per episode, others
     emit a single row group for the whole dataset.
  2. Normalizes each episode's schema to what v2.1's own meta/info.json
     declares — `fixed_size_list<float32>[N]` for the packed state/action
     vectors, float32 timestamp — and stamps the matching HuggingFace
     `datasets` features metadata. Some v3 writers store these as
     `list<double>` while still declaring float32 in info.json; carrying that
     mismatch forward makes the dataset disagree with its own metadata.
  3. Cuts each episode out of the aggregated source videos and re-encodes to
     h264/yuv420p, 640x360, crf 23, gop 10, exactly `length` frames per clip
     (hard-verified by decoding the output and counting).
  4. Writes v2.1 meta/: info.json, tasks.jsonl, episodes.jsonl, stats.json.

Camera mapping: DreamZero's yam recipe expects top/left/right cameras, so the
source camera keys are renamed to `observation.images.{top,left,right}
_camera-images-rgb`. The mapping is derived from the source's own video
features (a key containing "left"/"right" is that wrist; the remaining one —
ego, front, head, scene — is top) and PRINTED for you to check: camera order is
semantic, and getting it wrong trains a broken model without erroring. Override
with --camera-map when the heuristic can't tell.

Resolution choice: 640x360 for ALL cameras. OpenArms sources are 16:9
landscape, so a single 16:9 target introduces no letterboxing or distortion;
DreamZero's yam transform resizes to 320x176 at train time, so 640x360 keeps
~2x headroom while cutting storage and decode cost several-fold. Even
dimensions keep yuv420p happy. A source that isn't ~16:9 is flagged — pass
--out-size then.

ffmpeg seek strategy (frame accuracy): with re-encoding, `-ss` placed BEFORE
`-i` is frame-accurate in modern ffmpeg (the demuxer seeks to the nearest
keyframe at-or-before the target, then decodes forward and discards frames
whose pts is below the target). The trap is float rounding: from_timestamp is
stored as a float and can land an epsilon ABOVE the true first-frame pts, which
would silently drop that frame. We therefore seek to
(from_timestamp - 0.5/fps) -- strictly between the previous frame and the
wanted first frame -- and take exactly `-frames:v length` frames instead of
trusting `-to`. setpts rewrites output pts to a clean CFR timeline from 0.

Muxer gotcha (found empirically on ffmpeg 7.0.2): with libx264's default
B-frames the mp4 muxer writes an edit list (negative CTS shift), and demuxers
then drop the trailing frame -- a 460-packet file decodes to 459 frames. We
encode with `-bf 0` and mux with `-use_editlist 0`; verified this yields
exactly `length` decodable frames with start=0.0. No B-frames is also friendlier
for random-access clip decoding during training.

Usage:
  python3 pipeline/convert_lerobot_v3_to_v21.py <src_v3_dir> <output_dir> \
      [--episodes N] [--workers 8] [--camera-map old=new,...] [--out-size WxH]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = None  # set from the positional src argument in main()

# yam camera names the embodiment configs and DreamZero's recipe expect
YAM_CAMERAS = {
    "top": "observation.images.top_camera-images-rgb",
    "left": "observation.images.left_camera-images-rgb",
    "right": "observation.images.right_camera-images-rgb",
}
OUT_W, OUT_H = 640, 360
CRF = "23"
GOP = "10"
DATA_PATH_TPL = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TPL = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000  # datasets under 1000 episodes land entirely in chunk-000

# info.json dtype -> arrow type, for the schema normalization in step 2
ARROW_DTYPES = {
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "int32": pa.int32(),
    "bool": pa.bool_(),
}


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        sys.exit("ERROR: no ffmpeg found (imageio_ffmpeg missing and none on PATH)")


def count_video_frames(ffmpeg, path):
    """Count frames by fully decoding (no ffprobe available on this host)."""
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-progress", "pipe:1",
           "-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError("frame count failed for %s: %s" % (path, res.stderr[-500:]))
    frames = -1
    for line in res.stdout.splitlines():
        if line.startswith("frame="):
            frames = int(line.split("=", 1)[1].strip())
    if frames < 0:
        raise RuntimeError("could not parse frame count for %s" % path)
    return frames


def encode_episode_video(job):
    """Worker: cut one (episode, camera) clip. Returns (out_file, n_frames, elapsed, note)."""
    (ffmpeg, src_file, out_file, from_ts, length, fps, out_w, out_h) = job
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # resume: accept an existing output if its frame count already matches
    if out_file.exists():
        try:
            if count_video_frames(ffmpeg, out_file) == length:
                return (str(out_file), length, 0.0, "skipped (exists, frame count ok)")
        except Exception:
            pass
        out_file.unlink()

    vf = "scale=%d:%d:flags=bicubic,setpts=N/(%g*TB)" % (out_w, out_h, fps)
    seek = max(0.0, from_ts - 0.5 / fps)  # half-frame early: robust to float rounding of from_ts
    t0 = time.time()
    common_out = ["-frames:v", str(length), "-vf", vf, "-fps_mode", "passthrough",
                  "-c:v", "libx264", "-preset", "medium", "-crf", CRF, "-g", GOP,
                  "-bf", "0", "-pix_fmt", "yuv420p", "-threads", "4",
                  "-video_track_timescale", "15360", "-use_editlist", "0",
                  "-an", "-movflags", "+faststart", str(out_file)]
    attempts = [
        # 1) fast+accurate input seek
        [ffmpeg, "-y", "-v", "error", "-nostdin",
         "-ss", "%.6f" % seek, "-i", str(src_file)] + common_out,
        # 2) fallback: output-side seek (decode from start of source; slow but exact)
        [ffmpeg, "-y", "-v", "error", "-nostdin",
         "-i", str(src_file), "-ss", "%.6f" % seek] + common_out,
    ]
    last_err = ""
    for i, cmd in enumerate(attempts):
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            last_err = "ffmpeg rc=%d: %s" % (res.returncode, res.stderr[-500:])
            continue
        n = count_video_frames(ffmpeg, out_file)
        if n == length:
            note = "ok" if i == 0 else "ok (output-seek fallback)"
            return (str(out_file), n, time.time() - t0, note)
        last_err = "frame count %d != expected %d (attempt %d)" % (n, length, i + 1)
    raise RuntimeError("FAILED %s: %s" % (out_file, last_err))


def source_cameras(src_info):
    return [k for k, v in src_info["features"].items() if v.get("dtype") == "video"]


def resolve_camera_map(src_info, override):
    """source camera key -> yam camera key, from --camera-map or by name.

    Order is semantic (top = scene, left/right = wrist), and a wrong mapping
    trains a broken model silently, so the result is always printed and an
    ambiguous source is a hard error rather than a guess.
    """
    cams = source_cameras(src_info)
    if not cams:
        sys.exit("ERROR: source meta/info.json declares no video features")

    if override:
        cam_map = {}
        for pair in override.split(","):
            if "=" not in pair:
                sys.exit("ERROR: --camera-map wants old=new pairs, got %r" % pair)
            old, new = (s.strip() for s in pair.split("=", 1))
            if old not in cams:
                sys.exit("ERROR: --camera-map source key %r is not a video feature. "
                         "Available: %s" % (old, ", ".join(cams)))
            cam_map[old] = new
        return cam_map

    slots = {}
    for key in cams:
        name = key.rsplit(".", 1)[-1].lower()
        slot = "left" if "left" in name else "right" if "right" in name else "top"
        slots.setdefault(slot, []).append(key)
    if sorted(slots) != ["left", "right", "top"] or any(len(v) != 1 for v in slots.values()):
        sys.exit(
            "ERROR: could not map the source cameras %s onto yam's top/left/right\n"
            "       (matched: %s). Pass the mapping explicitly, e.g.\n"
            "       --camera-map '%s=%s,...'"
            % (cams, {k: v for k, v in slots.items()}, cams[0], YAM_CAMERAS["top"]))
    return {slots[slot][0]: YAM_CAMERAS[slot] for slot in ("top", "left", "right")}


def check_aspect(src_info, cam_map, out_w, out_h):
    """Warn when a source camera's aspect ratio differs from the output's."""
    want = out_w / out_h
    for key in cam_map:
        shape = src_info["features"][key].get("shape") or []
        if len(shape) >= 2 and shape[0]:
            got = shape[1] / shape[0]  # shape is [H, W, C]
            if abs(got - want) > 0.02:
                print("  WARNING: %s is %dx%d (aspect %.3f) but the output is "
                      "%dx%d (%.3f) — the clip will be distorted. Pass "
                      "--out-size to match." % (key, shape[1], shape[0], got,
                                                out_w, out_h, want))


def v21_schema(tbl, src_info):
    """The arrow schema v2.1's info.json declares, for the columns present.

    v2.1 datasets store packed vectors as fixed_size_list<float32>[N] with
    float32 scalars (verified on the validated ALOHA and OpenArms datasets).
    Columns info.json says nothing about are left exactly as they are.
    """
    fields, changed, hf_features = [], [], {}
    for name in tbl.schema.names:
        have = tbl.schema.field(name).type
        feat = src_info["features"].get(name) or {}
        dtype = feat.get("dtype")
        base = ARROW_DTYPES.get(dtype)
        shape = feat.get("shape") or []
        want = have
        if base is not None and len(shape) == 1:
            if shape[0] > 1:
                want = pa.list_(pa.field("element", base), shape[0])
                hf_features[name] = {"feature": {"dtype": dtype, "_type": "Value"},
                                     "length": shape[0], "_type": "Sequence"}
            else:
                want = base
                hf_features[name] = {"dtype": dtype, "_type": "Value"}
        fields.append(pa.field(name, want))
        if want != have:
            changed.append("%s: %s -> %s" % (name, have, want))
    # The HuggingFace `datasets` features blob the v2.1 datasets carry. Written
    # only when every column is described by info.json; otherwise a partial blob
    # would be worse than letting `datasets` infer from the arrow types.
    meta = None
    if len(hf_features) == len(fields):
        meta = {"huggingface": json.dumps({"info": {"features": hf_features}})}
    schema = pa.schema(fields, metadata=meta)
    return schema, changed


def episode_bounds(tbl, src_pq):
    """episode_index -> (start_row, n_rows), from runs of the episode_index column.

    Row groups are deliberately not used: writers differ on whether a row group
    is an episode. Interleaved (non-contiguous) episode rows would make a
    run-based split drop data, so that is checked, not assumed.
    """
    ep = tbl.column("episode_index").to_numpy()
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ep)) + 1))
    stops = np.concatenate((starts[1:], [len(ep)]))
    if len(starts) != len(np.unique(ep)):
        sys.exit("ERROR: %s interleaves episodes (%d runs for %d distinct "
                 "episode_index values) — this converter needs each episode's "
                 "rows contiguous and in order" % (src_pq, len(starts), len(np.unique(ep))))
    return {int(ep[s]): (int(s), int(e - s)) for s, e in zip(starts, stops)}


def load_episode_meta():
    """Concat meta/episodes parquets -> DataFrame sorted by episode_index."""
    files = sorted((SRC / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        sys.exit("ERROR: no meta/episodes parquets found under %s" % SRC)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def split_parquets(ep_meta, out_dir, n_episodes, src_info):
    """Split the aggregated v3 data parquet(s) into per-episode v2.1 parquets."""
    print("== Splitting data parquets (%d episodes) ==" % n_episodes)
    lengths = {}
    by_file = {}
    for _, row in ep_meta.iterrows():
        ep = int(row["episode_index"])
        if ep >= n_episodes:
            continue
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        by_file.setdefault(key, []).append((ep, int(row["length"])))

    for (chunk_idx, file_idx), eps in sorted(by_file.items()):
        src_pq = SRC / ("data/chunk-%03d/file-%03d.parquet" % (chunk_idx, file_idx))
        if not src_pq.exists():
            sys.exit("ERROR: missing source data file %s" % src_pq)
        tbl = pq.read_table(src_pq)
        bounds = episode_bounds(tbl, src_pq)
        schema, changed = v21_schema(tbl, src_info)
        if changed:
            print("  %s: normalizing to the dtypes info.json declares — %s"
                  % (src_pq.name, "; ".join(changed)))
            for f in schema:
                have = tbl.schema.field(f.name).type
                narrowing = (pa.types.is_fixed_size_list(f.type)
                             and pa.types.is_float32(f.type.value_type)
                             and (pa.types.is_list(have)
                                  or pa.types.is_large_list(have)
                                  or pa.types.is_fixed_size_list(have))
                             and pa.types.is_float64(have.value_type))
                if narrowing:
                    a = np.asarray(tbl.column(f.name).to_pylist(), dtype=np.float64)
                    err = float(np.abs(a - a.astype(np.float32)).max())
                    print("    %s: largest value change from the float32 cast %.3g"
                          % (f.name, err))
        tbl = tbl.cast(schema)

        for ep, meta_len in sorted(eps):
            if ep not in bounds:
                sys.exit("ERROR: episode %d is not in %s (episodes present: %d..%d)"
                         % (ep, src_pq, min(bounds), max(bounds)))
            start, n_rows = bounds[ep]
            sub = tbl.slice(start, n_rows)
            # --- verification ---
            ep_col = np.unique(sub.column("episode_index").to_numpy())
            frame_idx = sub.column("frame_index").to_numpy()
            ts0 = float(sub.column("timestamp").to_numpy()[0])
            errs = []
            if sub.num_rows != meta_len:
                errs.append("rows %d != meta length %d" % (sub.num_rows, meta_len))
            if len(ep_col) != 1 or int(ep_col[0]) != ep:
                errs.append("episode_index col %s != %d" % (ep_col, ep))
            if not np.array_equal(frame_idx, np.arange(sub.num_rows)):
                errs.append("frame_index is not 0..%d" % (sub.num_rows - 1))
            if ts0 != 0.0:
                errs.append("timestamp starts at %r" % ts0)
            # the v3 meta's own row offsets must agree with the split
            row = ep_meta.loc[ep_meta["episode_index"] == ep].iloc[0]
            if "dataset_from_index" in ep_meta.columns:
                want_rows = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
                if want_rows != n_rows:
                    errs.append("meta dataset_from/to_index span %d rows, split found %d"
                                % (want_rows, n_rows))
            if errs:
                sys.exit("ERROR: episode %d in %s: %s" % (ep, src_pq, "; ".join(errs)))
            dst = out_dir / DATA_PATH_TPL.format(episode_chunk=ep // CHUNKS_SIZE,
                                                 episode_index=ep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(sub, dst)
            lengths[ep] = meta_len
        print("  %s -> %d episode files" % (src_pq.name, len(eps)))
    return lengths


def build_video_jobs(ep_meta, out_dir, n_episodes, ffmpeg, fps, cam_map, out_w, out_h):
    jobs = []
    for _, row in ep_meta.iterrows():
        ep = int(row["episode_index"])
        if ep >= n_episodes:
            continue
        length = int(row["length"])
        for old_key, new_key in cam_map.items():
            col = "videos/%s/chunk_index" % old_key
            if col not in ep_meta.columns:
                sys.exit("ERROR: meta/episodes has no %r column — this v3 tree "
                         "does not record per-episode video offsets, so clips "
                         "cannot be cut. Available columns: %s"
                         % (col, [c for c in ep_meta.columns if c.startswith("videos/")]))
            c = int(row[col])
            f = int(row["videos/%s/file_index" % old_key])
            from_ts = float(row["videos/%s/from_timestamp" % old_key])
            to_ts = float(row["videos/%s/to_timestamp" % old_key])
            n_expect = round((to_ts - from_ts) * fps)
            if n_expect != length:
                sys.exit("ERROR: ep %d %s: (to-from)*fps=%d != length %d"
                         % (ep, old_key, n_expect, length))
            src_file = SRC / ("videos/%s/chunk-%03d/file-%03d.mp4" % (old_key, c, f))
            if not src_file.exists():
                sys.exit("ERROR: missing source video %s" % src_file)
            dst = out_dir / VIDEO_PATH_TPL.format(episode_chunk=ep // CHUNKS_SIZE,
                                                  video_key=new_key, episode_index=ep)
            jobs.append((ffmpeg, str(src_file), str(dst), from_ts, length, fps, out_w, out_h))
    return jobs


def read_tasks(src):
    """[{task_index, task}] from meta/tasks.parquet, whichever shape it has.

    v3 writers differ: some store `task` as the index with a `task_index`
    column, others store both as columns. Reading the wrong one writes the
    integer index as the language annotation and trains on the string "0".
    """
    df = pd.read_parquet(src / "meta" / "tasks.parquet")
    if "task" in df.columns and "task_index" in df.columns:
        pairs = zip(df["task_index"], df["task"])
    elif "task_index" in df.columns and df.index.name == "task":
        pairs = zip(df["task_index"], df.index)
    elif "task" in df.columns and df.index.name == "task_index":
        pairs = zip(df.index, df["task"])
    else:
        sys.exit("ERROR: cannot read meta/tasks.parquet — expected task/task_index "
                 "as columns or index, got index %r and columns %s"
                 % (df.index.name, list(df.columns)))
    return sorted(({"task_index": int(ti), "task": str(t)} for ti, t in pairs),
                  key=lambda d: d["task_index"])


def write_meta(ep_meta, out_dir, n_episodes, total_frames, src_info, fps, cam_map,
               out_w, out_h):
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ---- tasks.jsonl
    task_entries = read_tasks(SRC)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in task_entries:
            f.write(json.dumps(t) + "\n")

    # ---- episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for _, row in ep_meta.iterrows():
            ep = int(row["episode_index"])
            if ep >= n_episodes:
                continue
            f.write(json.dumps({
                "episode_index": ep,
                "tasks": [str(t) for t in row["tasks"]],
                "length": int(row["length"]),
            }) + "\n")

    # ---- info.json (v2.1)
    features = {}
    for key, feat in src_info["features"].items():
        if key in cam_map:
            new_feat = json.loads(json.dumps(feat))  # deep copy
            new_feat["shape"] = [out_h, out_w, 3]
            vinfo = new_feat.get("info", {})
            vinfo["video.height"] = out_h
            vinfo["video.width"] = out_w
            vinfo["video.codec"] = "h264"
            vinfo["video.pix_fmt"] = "yuv420p"
            vinfo["video.fps"] = fps
            new_feat["info"] = vinfo
            features[cam_map[key]] = new_feat
        elif key in source_cameras(src_info):
            continue  # camera not in the map: dropped from the v2.1 dataset
        else:
            features[key] = feat
    info = {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "unknown"),
        "total_episodes": n_episodes,
        "total_frames": int(total_frames),
        "total_tasks": len(task_entries),
        "total_videos": n_episodes * len(cam_map),
        "total_chunks": (n_episodes - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": "0:%d" % n_episodes},
        "data_path": DATA_PATH_TPL,
        "video_path": VIDEO_PATH_TPL,
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # ---- stats.json (copy, rename video feature keys)
    stats_src = SRC / "meta" / "stats.json"
    if stats_src.exists():
        with open(stats_src) as f:
            stats = json.load(f)
        stats = {cam_map.get(k, k): v for k, v in stats.items()}
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(stats, f)
    return task_entries


def main():
    global SRC
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="source LeRobot v3.0 dataset dir")
    ap.add_argument("output_dir", type=Path, help="where to write the v2.1 dataset")
    ap.add_argument("--episodes", type=int, default=None,
                    help="only convert the first N episodes (quick test)")
    ap.add_argument("--workers", type=int, default=8,
                    help="ffmpeg process pool size (default 8)")
    ap.add_argument("--camera-map", default=None,
                    help="explicit source=target camera pairs, comma-separated, "
                         "when the name heuristic can't map them onto yam's "
                         "top/left/right")
    ap.add_argument("--out-size", default="%dx%d" % (OUT_W, OUT_H),
                    help="output clip size WxH (default %dx%d)" % (OUT_W, OUT_H))
    args = ap.parse_args()

    try:
        out_w, out_h = (int(v) for v in args.out_size.lower().split("x"))
    except ValueError:
        sys.exit("ERROR: --out-size wants WxH, e.g. 640x360 (got %r)" % args.out_size)
    if out_w % 2 or out_h % 2:
        sys.exit("ERROR: --out-size dimensions must be even for yuv420p (got %dx%d)"
                 % (out_w, out_h))

    SRC = args.src.resolve()
    if not (SRC / "meta" / "info.json").exists():
        sys.exit("ERROR: %s has no meta/info.json" % SRC)
    out_dir = args.output_dir.resolve()
    if out_dir == SRC or str(out_dir).startswith(str(SRC) + os.sep):
        sys.exit("ERROR: output_dir must not be inside the source dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    print("ffmpeg: %s" % ffmpeg)

    with open(SRC / "meta" / "info.json") as f:
        src_info = json.load(f)
    if src_info.get("codebase_version") != "v3.0":
        sys.exit("ERROR: expected a v3.0 source, got %s — a v2.1 dataset needs no "
                 "conversion" % src_info.get("codebase_version"))
    fps = float(src_info["fps"])
    ep_meta = load_episode_meta()
    total_available = len(ep_meta)
    n_episodes = min(args.episodes, total_available) if args.episodes else total_available

    cam_map = resolve_camera_map(src_info, args.camera_map)
    print("source:  %d episodes at %g fps; converting %d" % (total_available, fps, n_episodes))
    print("cameras: (check this — order is semantic and a wrong map trains a "
          "broken model silently)")
    for old, new in cam_map.items():
        print("    %-40s -> %s" % (old, new))
    dropped = [c for c in source_cameras(src_info) if c not in cam_map]
    if dropped:
        print("    dropped (not in the map): %s" % ", ".join(dropped))
    check_aspect(src_info, cam_map, out_w, out_h)

    # 1) parquet split + verification
    lengths = split_parquets(ep_meta, out_dir, n_episodes, src_info)
    total_frames = sum(lengths.values())

    # 2) videos
    jobs = build_video_jobs(ep_meta, out_dir, n_episodes, ffmpeg, fps, cam_map,
                            out_w, out_h)
    print("== Encoding %d video clips (%d workers, %dx%d h264 crf %s gop %s) =="
          % (len(jobs), args.workers, out_w, out_h, CRF, GOP))
    done, failures, fallbacks, skipped = 0, [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(encode_episode_video, j): j for j in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                out_file, n, dt, note = fut.result()
                done += 1
                if "fallback" in note:
                    fallbacks += 1
                if "skipped" in note:
                    skipped += 1
                if done % 25 == 0 or done == len(jobs):
                    print("  [%d/%d] %s (%d frames, %.1fs, %s)"
                          % (done, len(jobs), Path(out_file).name, n, dt, note))
            except Exception as e:
                failures.append((job[2], str(e)))
                print("  FAIL %s: %s" % (job[2], e), file=sys.stderr)
    if failures:
        sys.exit("ERROR: %d/%d video clips failed frame-exact encoding -- aborting "
                 "before meta write" % (len(failures), len(jobs)))

    # 3) meta
    task_entries = write_meta(ep_meta, out_dir, n_episodes, total_frames, src_info,
                              fps, cam_map, out_w, out_h)

    # 4) summary
    print("\n== VALIDATION SUMMARY ==")
    print("output:            %s" % out_dir)
    print("episodes:          %d (row count, frame_index, timestamp, episode_index "
          "and meta row offsets all checked)" % n_episodes)
    print("total_frames:      %d" % total_frames)
    print("video clips:       %d encoded, frame counts verified == episode length" % len(jobs))
    print("                   (%d via output-seek fallback, %d resumed/skipped)"
          % (fallbacks, skipped))
    print("cameras:           %s, %dx%d h264/yuv420p"
          % (", ".join("%s->%s" % (o.rsplit(".", 1)[-1], n.rsplit(".", 1)[-1].split("_")[0])
                       for o, n in cam_map.items()), out_w, out_h))
    print("tasks:             %s" % json.dumps(task_entries))
    print("meta files:        info.json (v2.1), tasks.jsonl, episodes.jsonl, stats.json")
    print("\n== NEXT STEP ==")
    print("Point project_config.json's dataset.local_path at\n"
          "  %s\n"
          "and check that your embodiment config's fps says %g, then run\n"
          "  python3 pipeline/run_pipeline.py --name <run-name>\n"
          "— the pipeline's validate and prep stages take it from here (the GEAR\n"
          "conversion uses your embodiment config)." % (out_dir, fps))


if __name__ == "__main__":
    main()
