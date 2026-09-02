#!/usr/bin/env python3
"""Tests for prep_dataset.py's codec detection + re-encode branch.

Real LeRobot v2.1 datasets nest a camera's codec metadata under one of two
keys depending on provenance: hub datasets converted from v1.x (e.g.
lerobot/aloha_* at rev v2.1) use "video_info"; datasets recorded by newer
lerobot versions (and v3-style trees) use "info". prep_dataset.py must
detect the codec — and rewrite it after re-encoding — under either key,
because missing it does not error: the videos just stay in a codec the
training container cannot rely on decoding.

Each case builds a tiny synthetic v2.1 dataset (a real 5-frame h264 clip via
imageio-ffmpeg, dummy parquet, minimal meta) plus a stub DreamZero repo whose
convert_lerobot_to_gear.py only writes modality.json, then runs
prep_dataset.py as a subprocess exactly as the pipeline does. The "av1"
declarations are metadata-only (the actual bytes are h264) so the re-encode
branch runs without needing an av1 encoder in the test environment.

Run:  python3 pipeline/tests/test_prep_dataset.py   (or pytest)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREP = HERE.parent / "prep_dataset.py"


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _make_dataset(root: Path, meta_key: str, codec: str) -> Path:
    """A minimal v2.1 LeRobot tree: 1 episode, 1 camera, real h264 video."""
    src = root / "src"
    (src / "data" / "chunk-000").mkdir(parents=True)
    (src / "meta").mkdir()
    cam_dir = src / "videos" / "chunk-000" / "observation.images.cam0"
    cam_dir.mkdir(parents=True)
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=0.5:size=64x64:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(cam_dir / "episode_000000.mp4")],
        check=True)
    (src / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(
        b"not read by prep")  # prep copies data/ verbatim; only meta matters
    (src / "meta" / "tasks.jsonl").write_text(
        '{"task_index": 0, "task": "test"}\n')
    (src / "meta" / "episodes.jsonl").write_text(
        '{"episode_index": 0, "length": 5}\n')
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_videos": 1,
        "fps": 10,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.images.cam0": {
                "dtype": "video", "shape": [64, 64, 3],
                meta_key: {"video.codec": codec, "video.pix_fmt": "yuv420p",
                           "video.fps": 10.0},
            },
        },
    }
    json.dump(info, open(src / "meta" / "info.json", "w"), indent=4)
    return src


def _make_stub_dreamzero(root: Path) -> Path:
    """Stand-in for the DreamZero clone: the converter only writes modality.json."""
    stub = root / "dzstub"
    conv = stub / "scripts" / "data"
    conv.mkdir(parents=True)
    (conv / "convert_lerobot_to_gear.py").write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--dataset-path', required=True)\n"
        "args, _ = ap.parse_known_args()\n"
        "json.dump({'state': {}, 'action': {}, 'video': {}},\n"
        "          open(Path(args.dataset_path) / 'meta' / 'modality.json', 'w'))\n")
    return stub


def _make_config(root: Path, reencode_codecs) -> Path:
    cfg = root / "embodiment.yaml"
    cfg.write_text(
        "embodiment_tag: yam\n"
        "camera_map:\n  cam0: top_camera-images-rgb\n"
        "state_keys:\n  joints: [0, 2]\n"
        "action_keys:\n  joints: [0, 2]\n"
        "relative_action_keys:\n  - joints\n"
        "task_key: task_index\n"
        "action_horizon: 24\n"
        "fps: 10\n"
        f"reencode_codecs: {json.dumps(list(reencode_codecs))}\n")
    return cfg


def _run_prep(root: Path, src: Path, cfg: Path) -> Path:
    dst = root / "dst"
    env = dict(os.environ, DREAMZERO_REPO=str(_make_stub_dreamzero(root)))
    subprocess.run(
        [sys.executable, str(PREP), "--src", str(src), "--dst", str(dst),
         "--config", str(cfg), "--workers", "2"],
        check=True, env=env, cwd=str(root))
    return dst


def _case(meta_key: str, declared_codec: str, reencode: bool):
    with tempfile.TemporaryDirectory(prefix="prep_test_") as td:
        root = Path(td)
        src = _make_dataset(root, meta_key, declared_codec)
        cfg = _make_config(root, ["av1", "av01"])
        dst = _run_prep(root, src, cfg)

        src_mp4 = (src / "videos" / "chunk-000" / "observation.images.cam0"
                   / "episode_000000.mp4").read_bytes()
        out_path = (dst / "videos" / "chunk-000"
                    / "observation.images.top_camera-images-rgb"
                    / "episode_000000.mp4")
        assert out_path.exists(), f"missing output video: {out_path}"
        out_mp4 = out_path.read_bytes()

        feat = json.load(open(dst / "meta" / "info.json"))["features"][
            "observation.images.top_camera-images-rgb"]
        assert meta_key in feat and isinstance(feat[meta_key], dict), (
            f"metadata key {meta_key!r} lost in the rewrite: {feat}")
        other = {"info": "video_info", "video_info": "info"}[meta_key]
        assert other not in feat, f"rewrite invented a {other!r} key: {feat}"

        if reencode:
            assert out_mp4 != src_mp4, (
                f"[{meta_key}/{declared_codec}] video was copied, not "
                f"re-encoded — the re-encode branch did not fire")
            assert feat[meta_key]["video.codec"] == "h264", (
                f"[{meta_key}] codec not rewritten to h264: {feat[meta_key]}")
        else:
            assert out_mp4 == src_mp4, (
                f"[{meta_key}/{declared_codec}] video was re-encoded but the "
                f"codec is not in reencode_codecs")
            assert feat[meta_key]["video.codec"] == declared_codec

        mod = json.load(open(dst / "meta" / "modality.json"))
        assert mod["annotation"] == {"task": {"original_key": "task_index"}}


def test_reencode_fires_for_info_key():
    # newer-LeRobot layout: codec under "info" — the case that was silently
    # skipped when detect_codec read only "video_info"
    _case("info", "av1", reencode=True)


def test_reencode_fires_for_video_info_key():
    # hub-v2.1 layout (lerobot/aloha_* — the golden-test dataset)
    _case("video_info", "av1", reencode=True)


def test_h264_is_copied_untouched():
    _case("info", "h264", reencode=False)


if __name__ == "__main__":
    for fn in (test_reencode_fires_for_info_key,
               test_reencode_fires_for_video_info_key,
               test_h264_is_copied_untouched):
        fn()
        print(f"PASS {fn.__name__}")
    print("all prep_dataset tests passed")
