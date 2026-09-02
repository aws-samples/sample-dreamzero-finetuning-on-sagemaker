#!/usr/bin/env python3
"""Turn deployed CDK stack outputs into pipeline/pipeline_config.json.

Run after `cdk deploy`:
  python3 generate_pipeline_config.py [--stack dreamzero-pipeline-infra] \
      [--region us-east-1] [--profile my-profile] [--out path.json]

Reads the stack's CloudFormation outputs and writes the JSON the pipeline
scripts consume. No values are hardcoded — everything comes from the account
the stack was deployed into.

Pass --region when the stack is not in the region your ambient credentials
resolve to. Note that boto3 does NOT read AWS_REGION (only AWS_DEFAULT_REGION,
then the profile's own region), so a shell set up for `cdk deploy` with
AWS_REGION alone will send this script somewhere else entirely — which, if you
have deployed the stack into two regions, silently succeeds against the wrong
one. `./setup.sh` forwards --stack/--region/--profile here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from solution import get_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pipeline" / "pipeline_config.json"


def _configured_tag() -> str:
    """image.tag from the repo-root project_config.json — the tag the
    deploy-time kickoff build pushes; fallback before any build published a
    digest-pinned URI to SSM."""
    try:
        return (json.loads((ROOT / "project_config.json").read_text())
                .get("image", {}).get("tag")) or "v11"
    except (OSError, json.JSONDecodeError):
        return "v11"

# CfnOutput logical id -> config key
MAP = {
    "BucketName": "bucket",
    "EcrRepoUri": "ecr_repo_uri",
    "SageMakerRoleArn": "sagemaker_role_arn",
    "ServingInstanceProfileName": "serving_instance_profile",
    "Region": "region",
    "Account": "account",
    "Project": "project",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="dreamzero-pipeline-infra")
    ap.add_argument("--region", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--image-tag", default=None,
                    help="tag of the BYOC image in the ECR repo. Default: the "
                         "digest-pinned URI the CodeBuild image factory wrote "
                         "to SSM (/dreamzero/<project>/image_uri); if no build "
                         "has finished yet, falls back to project_config.json "
                         "image.tag")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="print a one-line summary instead of the full config "
                         "(what setup.sh uses)")
    ap.add_argument("--out", default=os.environ.get("DREAMZERO_PIPELINE_CONFIG")
                                    or str(OUT),
                    help="where to write the config. Default: "
                         "$DREAMZERO_PIPELINE_CONFIG, else "
                         "pipeline/pipeline_config.json — the path "
                         "pipeline_config.py loads by default. Give each region "
                         "its own file if you deploy the stack more than once.")
    args = ap.parse_args()
    out = Path(args.out)

    sess = boto3.Session(profile_name=args.profile, region_name=args.region)
    cfn = get_client("cloudformation", session=sess)
    outs = cfn.describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]
    raw = {o["OutputKey"]: o["OutputValue"] for o in outs}

    cfg = {MAP[k]: v for k, v in raw.items() if k in MAP}
    if args.image_tag:
        cfg["image_uri"] = f"{cfg['ecr_repo_uri']}:{args.image_tag}"
    else:
        # The image factory publishes the digest-pinned URI after each
        # successful build; prefer it — the ECR repo has MUTABLE tags, so a
        # tag reference could change under a running multi-day job.
        param = f"/dreamzero/{cfg.get('project', 'dreamzero')}/image_uri"
        try:
            cfg["image_uri"] = get_client("ssm", session=sess).get_parameter(
                Name=param)["Parameter"]["Value"]
            print(f"image_uri from SSM {param} (digest-pinned)")
        except Exception as e:
            tag = _configured_tag()
            cfg["image_uri"] = f"{cfg['ecr_repo_uri']}:{tag}"
            print(f"note: no image build published to SSM yet "
                  f"({type(e).__name__}) — defaulting to :{tag}. The deploy "
                  f"kicked one off; re-run this once it finishes (watch it in "
                  f"CodeBuild, see cdk/README.md), or pass --image-tag")
    cfg["s3_root"] = f"s3://{cfg['bucket']}/sagemaker"
    if args.profile:
        cfg["profile"] = args.profile

    # Neither of these comes from stack outputs, so a plain regenerate would
    # drop them. extra_job_tags is the one that actually matters: it has no CLI
    # flag and no env var, so dropping it is unrecoverable and leaves the next
    # queued job eligible for being stopped by account cleanup tooling.
    # profile is carried for convenience only — it can also come from --profile
    # or DREAMZERO_AWS_PROFILE, and dropping it is close to harmless, since
    # regeneration can only succeed if the ambient credential chain already
    # reaches the stack, and every consumer falls back to that same chain.
    # An explicit --profile wins.
    if out.exists():
        try:
            prev = json.loads(out.read_text())
        except json.JSONDecodeError:
            prev = {}
        for key in ("extra_job_tags", "profile"):
            if prev.get(key) and not cfg.get(key):
                cfg[key] = prev[key]
                print(f"carried over {key}: {cfg[key]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    img = cfg.get("image_uri", "")
    shown = (f"@{img.split('@sha256:')[1][:12]}… (digest-pinned)"
             if "@sha256:" in img else f":{img.rsplit(':', 1)[-1]} (tag)")
    # The region is echoed because it is the value most likely to be wrong (see
    # the module docstring) and the stack it came from is the only proof of it.
    print(f"wrote {out}: stack {args.stack}, bucket {cfg.get('bucket')}, "
          f"region {cfg.get('region')}, image {shown}")
    if not args.quiet:
        print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
