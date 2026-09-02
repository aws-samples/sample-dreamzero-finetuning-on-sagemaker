#!/usr/bin/env python3
"""One-time: download the DreamZero base assets from HuggingFace and stage to
the S3 bucket created by the CDK stack (~128GB / ~119GiB, downloaded then
uploaded — one figure, two unit systems; the README quotes the same number).

Reads the bucket from pipeline_config.json. umt5-xxl: tokenizer files only —
the T5 encoder weights come from Wan2.1's .pth (49GB saved).

Usage: python3 stage_base_assets.py [--staging-dir ~/dreamzero_staging]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from botocore.exceptions import ClientError

from pipeline_config import load_config

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Revisions are pinned to commit SHAs (2026-08-22) so a re-staging always
# fetches the exact weights this sample was validated against — an upstream
# force-push or compromised repo can silently change `main`, and the failure
# mode would be a subtly different base model, not an error. To move to newer
# upstream weights, update the SHA deliberately and re-validate.
JOBS = [
    ("GEAR-Dreams/DreamZero-AgiBot", "model",
     "a8e108456e537ab30e7e84523d41d12a7e739ca5",
     "checkpoints/DreamZero-AgiBot", None),
    ("Wan-AI/Wan2.1-I2V-14B-480P", "model",
     "6b73f84e66371cdfe870c72acd6826e1d61cf279",
     "checkpoints/Wan2.1-I2V-14B-480P", None),
    ("google/umt5-xxl", "model",
     "66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
     "checkpoints/umt5-xxl",
     ["*.json", "spiece.model", "*.txt", "*.md"]),
]

S3_DEST = {
    "checkpoints/DreamZero-AgiBot": "checkpoints/DreamZero-AgiBot",
    "checkpoints/Wan2.1-I2V-14B-480P": "checkpoints/Wan2.1-I2V-14B-480P",
    "checkpoints/umt5-xxl": "checkpoints/umt5-xxl-tokenizer",
}

# --check floors. A prefix must clear BOTH to count as staged: a killed sync
# leaves a partial prefix, and "non-empty" would wave it through — the missing
# shards then fail a $20+/hr training job at download time instead of here.
# Measured against a fully staged bucket. check_staged() counts EVERY object
# under the prefix, so the object totals include the .cache/huggingface/
# download metadata snapshot_download leaves behind; the payload counts are what
# a floor can safely require, because a stage done by some other route (a plain
# `aws s3 sync` of someone else's copy) has the payload but no .cache. Sizes are
# exact rather than approximate: every JOBS row pins a commit, so a complete
# prefix is byte-identical every time.
#
#   dest                            payload   total   bytes
#   DreamZero-AgiBot                     17      52   42.70 GiB
#   Wan2.1-I2V-14B-480P                  33     100   76.62 GiB
#   umt5-xxl-tokenizer                    8      25   20.48 MiB
#
# The byte floor is the load-bearing one — a partial prefix can still clear a
# count floor once the .cache metadata lands, so keep both near reality. The
# earlier Wan floor of (3, 10 GiB) passed a prefix holding 13% of the data.
CHECK_FLOORS = {  # dest -> (min objects, min bytes)
    "checkpoints/DreamZero-AgiBot": (17, 40 * 2**30),
    "checkpoints/Wan2.1-I2V-14B-480P": (33, 72 * 2**30),
    "checkpoints/umt5-xxl-tokenizer": (8, 15 * 2**20),
}


def check_staged(cfg, quiet=False):
    """True iff every base-asset prefix in S3 clears its size/count floor."""
    from pipeline_config import boto_session
    from solution import get_client
    s3 = get_client("s3", session=boto_session(cfg))
    bucket = cfg["bucket"]
    root = cfg["s3_root"].split(f"{bucket}/", 1)[1]
    ok = True
    for dest, (min_n, min_b) in CHECK_FLOORS.items():
        n = tot = 0
        try:
            for page in s3.get_paginator("list_objects_v2").paginate(
                    Bucket=bucket, Prefix=f"{root}/{dest}/"):
                for o in page.get("Contents", []):
                    n += 1
                    tot += o["Size"]
        except ClientError as e:
            # A raw botocore traceback here is actively misleading: the usual
            # cause is not a broken bucket but credentials resolving to the wrong
            # ACCOUNT, and setup.sh calls this as its last step, so the reader's
            # takeaway from 20 lines of stack was "the tool is broken" rather
            # than "check who you are". Note AccessDenied is also what a
            # correctly-scoped-but-different account returns, so it cannot be
            # distinguished from a genuine permission gap here.
            code = e.response.get("Error", {}).get("Code", "Unknown")
            print(f"  ERROR   {dest}: cannot list s3://{bucket}/{root}/{dest}/ "
                  f"({code})")
            if code in ("AccessDenied", "AllAccessDisabled", "InvalidAccessKeyId",
                        "ExpiredToken", "UnrecognizedClientException"):
                print(f"          The bucket in this config belongs to account "
                      f"{cfg.get('account', '<unset>')} in "
                      f"{cfg.get('region', '<unset>')}. Check that your "
                      f"credentials resolve there — `aws sts get-caller-identity`"
                      f" — and remember boto3 ignores AWS_REGION.")
            elif code == "NoSuchBucket":
                print(f"          That bucket does not exist. Re-run "
                      f"cdk/generate_pipeline_config.py against the stack you "
                      f"actually deployed.")
            return False
        good = n >= min_n and tot >= min_b
        ok = ok and good
        if not quiet:
            print(f"  {'staged ' if good else 'MISSING'} {dest}: "
                  f"{n} objects, {tot / 2**30:.2f} GiB")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging-dir", default=os.path.expanduser("~/dreamzero_staging"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="report what is already staged in S3 and exit "
                         "(0 = fully staged, 1 = something missing); "
                         "downloads nothing")
    args = ap.parse_args()

    cfg = load_config()
    if args.check:
        sys.exit(0 if check_staged(cfg) else 1)
    base = Path(args.staging_dir)

    if not args.skip_download:
        from huggingface_hub import snapshot_download
        for repo_id, repo_type, rev, subdir, patterns in JOBS:
            print(f"==> {repo_id} -> {base / subdir}", flush=True)
            snapshot_download(repo_id=repo_id, repo_type=repo_type,
                              revision=rev,  # nosec B615 — every JOBS row
                              # pins a commit SHA (see the table above);
                              # bandit can't see through the loop variable
                              local_dir=str(base / subdir), allow_patterns=patterns,
                              max_workers=16)

    for local, dest in S3_DEST.items():
        uri = f"{cfg['s3_root']}/{dest}"
        src = base / local
        n_files = tot = 0
        for p in src.rglob("*"):
            if p.is_file():
                n_files += 1
                tot += p.stat().st_size
        print(f"==> sync {local} ({n_files} files, {tot / 2**30:.1f} GiB) "
              f"-> {uri}", flush=True)
        # no --only-show-errors: a multi-GB upload should show the CLI's live
        # progress meter, not run silent for an hour
        cmd = ["aws", "s3", "sync", str(src), uri]
        if cfg.get("profile"):
            cmd += ["--profile", cfg["profile"]]
        subprocess.run(cmd, check=True)

    print("BASE ASSETS STAGED. Datasets are fetched per-run by run_pipeline.py "
          "(from HuggingFace via project_config.json / --hf-dataset, or a "
          "local path via --dataset).")


if __name__ == "__main__":
    main()
