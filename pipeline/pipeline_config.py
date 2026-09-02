"""Resolve pipeline infrastructure config (bucket, ECR image, roles, region).

Per-key precedence (highest first):
  1. individual DREAMZERO_* environment variables (override everything)
  2. the config file — an explicit path via $DREAMZERO_PIPELINE_CONFIG or
     load_config(path=...), else pipeline_config.json next to this file
     (written by generate_pipeline_config.py from the CDK stack outputs)
  3. SSM Parameter Store, /dreamzero/<project>/* — written by the CDK stack
     (bucket, s3_root, sagemaker_role_arn, region, project) and by its
     CodeBuild image factory (image_uri, digest-pinned, updated per build).
     Consulted only for keys the layers above left unset, so it costs no AWS
     call when a complete local config exists.

This is the single source of truth for account-specific values, so no runtime
script hardcodes an account id, bucket, profile, or role ARN.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from solution import get_client

_DEFAULT = Path(__file__).resolve().parent / "pipeline_config.json"

_REQUIRED = ["bucket", "region", "image_uri", "sagemaker_role_arn"]


def _fill_from_ssm(cfg: dict) -> dict:
    """Fill missing keys from the CDK stack's SSM parameters.

    The project name selects the parameter path (/dreamzero/<project>/):
    cfg["project"], then $DREAMZERO_PROJECT, then the default "dreamzero".
    Region/profile for the lookup come from whatever is already resolved,
    falling back to the ambient AWS config chain. Values already set locally
    always win — SSM only fills gaps. Any failure (no boto3, no credentials,
    no region, stack not deployed) degrades to a note; the caller still
    reports exactly which keys are missing.
    """
    project = (cfg.get("project") or os.environ.get("DREAMZERO_PROJECT")
               or "dreamzero")
    prefix = f"/dreamzero/{project}"
    try:
        import boto3
        kw = {}
        if cfg.get("profile"):
            kw["profile_name"] = cfg["profile"]
        if cfg.get("region"):
            kw["region_name"] = cfg["region"]
        ssm = get_client("ssm", session=boto3.Session(**kw))
        found = {}
        for page in ssm.get_paginator("get_parameters_by_path").paginate(
                Path=prefix):
            for p in page["Parameters"]:
                found[p["Name"].rsplit("/", 1)[-1]] = p["Value"]
    except Exception as e:  # noqa: BLE001 — any failure means "no SSM layer"
        print(f"note: SSM config fallback unavailable "
              f"({type(e).__name__}: {e}) — looked for {prefix}/*",
              file=sys.stderr)
        return cfg
    # s3_root is derived from bucket (see load_config's setdefault); taking it
    # from SSM could pair the stack's s3_root with a locally-set bucket — an
    # inconsistent combination run_pipeline.py only hard-aborts on at merge
    # time, after the training spend. When bucket itself comes from SSM the
    # derived value is identical anyway.
    filled = sorted(k for k, v in found.items()
                    if v and not cfg.get(k) and k != "s3_root")
    for k in filled:
        cfg[k] = found[k]
    if filled:
        print(f"config: {filled} resolved from SSM {prefix}/", file=sys.stderr)
    return cfg


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("DREAMZERO_PIPELINE_CONFIG") or str(_DEFAULT)
    cfg: dict = {}
    if Path(path).exists():
        cfg = json.load(open(path))

    # env overrides / fills
    env_map = {
        "bucket": "DREAMZERO_BUCKET",
        "region": "DREAMZERO_REGION",
        "image_uri": "DREAMZERO_IMAGE_URI",
        "sagemaker_role_arn": "DREAMZERO_SAGEMAKER_ROLE",
        "serving_instance_profile": "DREAMZERO_SERVING_PROFILE",
        "profile": "DREAMZERO_AWS_PROFILE",  # optional local named profile
    }
    for key, env in env_map.items():
        if os.environ.get(env):
            cfg[key] = os.environ[env]

    if any(not cfg.get(k) for k in _REQUIRED):
        cfg = _fill_from_ssm(cfg)

    missing = [k for k in _REQUIRED if not cfg.get(k)]
    if missing:
        raise SystemExit(
            f"pipeline config missing {missing}. Deploy the CDK stack, then "
            f"either run cdk/generate_pipeline_config.py, or set the "
            f"DREAMZERO_* env vars, or rely on the stack's SSM parameters "
            f"(/dreamzero/<project>/* — needs credentials + region; note "
            f"image_uri appears there only after the first CodeBuild image "
            f"build). (looked in {path})")

    cfg.setdefault("s3_root", f"s3://{cfg['bucket']}/sagemaker")
    return cfg


def job_tags(cfg: dict, workstream: str = "pipeline") -> list[dict]:
    """Tags for every SageMaker job this repo creates.

    `extra_job_tags` in pipeline_config.json is a passthrough for tags your
    account requires. Shared and sandbox accounts often run cost-cleanup tooling
    that stops or deletes resources on a schedule unless they carry an exemption
    tag — and a job queued for scarce GPU capacity is a prime target, because it
    can sit in Pending for hours. Such a job ends `Stopped` with no
    FailureReason at all, which is indistinguishable from a manual cancel.

    Every job-creating path in this repo must use this, not a hand-built list:
    an eval or merge job that misses the exemption tag gets stopped just as
    readily as a training job.
    """
    tags = [{"Key": "project", "Value": cfg.get("project", "dreamzero")},
            {"Key": "workstream", "Value": workstream}]
    extra = cfg.get("extra_job_tags") or {}
    if not isinstance(extra, dict):
        raise SystemExit(
            f"extra_job_tags must be a JSON object of key -> value, got "
            f"{type(extra).__name__}. Example: "
            f'{{"auto-stop": "no", "auto-delete": "never"}}')
    if not extra:
        # Loud, not fatal: on a personal account this is fine; on a shared
        # one it is the most expensive silent failure this repo has — a
        # multi-day job stopped mid-queue. Warn at submit time because the
        # config is read once at import, so a tag added to the file after
        # the driver started never reaches an already-running invocation.
        print(f"WARNING: extra_job_tags is empty — the {workstream} job "
              f"carries no cleanup-exemption tag. On a shared/sandbox "
              f"account, scheduled cleanup tooling may stop it (it would "
              f"end Stopped with no FailureReason). See pipeline/README.md "
              f"Caveats.", file=sys.stderr)
    tags += [{"Key": str(k), "Value": str(v)} for k, v in extra.items()]
    return tags


def boto_session(cfg: dict):
    """A boto3 Session on the resolved profile/region.

    Build clients from it with solution.get_client(..., session=...), never
    session.client(...) directly: the solution user-agent suffix is attached at
    client-construction time, so a bare .client() call is silently unattributed.
    """
    import boto3
    kw = {"region_name": cfg["region"]}
    if cfg.get("profile"):
        kw["profile_name"] = cfg["profile"]
    return boto3.Session(**kw)


# ---------------------------------------------------------------------------
# Per-run project config (dataset source, compute, hyperparameters).
# Account infra stays above; per-robot semantics stay in configs/*.yaml.
# project_config.json also carries the release-level "solution" block (id,
# name, version) that solution.py reads — it is not a per-run knob, has no
# entry in the defaults below, and solution.py deliberately reads the file at
# its fixed path rather than any --project-config override.
# ---------------------------------------------------------------------------

_PROJECT_DEFAULTS = {
    # relative local_path/cache_dir resolve against the repo root (the
    # repo's dataset/ folder is the gitignored home for local data)
    "dataset": {"source": "local", "repo_id": None, "revision": None,
                "local_path": None, "cache_dir": "dataset"},
    "embodiment_config": None,
    "validation": {"enabled": True, "fail_on_action_equals_state": False,
                   "degeneracy_sample_rows": 2000},
    "compute": {
        "smoke": {"instance_type": "ml.g7e.24xlarge", "volume_gb": 500,
                  "max_runtime_hours": 3, "steps": 10},
        "train": {"instance_type": "ml.g7e.24xlarge", "volume_gb": 500,
                  "max_runtime_hours": 12},
        "merge": {"instance_type": "ml.g7e.24xlarge", "volume_gb": 500,
                  "max_runtime_hours": 3},
        "eval": {"instance_type": "ml.g7e.24xlarge", "max_runtime_hours": 6},
    },
    "training": {"max_steps": 1000, "save_steps": 500,
                 "learning_rate": "1e-5",
                 "per_device_train_batch_size": 1, "seed": 42,
                 "warmup_ratio": 0.05},
}


def load_project_config(path: str | None = None) -> dict:
    """project_config.json over built-in defaults (two levels of dict merge)."""
    proj = {k: ({k2: (dict(v2) if isinstance(v2, dict) else v2)
                 for k2, v2 in v.items()} if isinstance(v, dict) else v)
            for k, v in _PROJECT_DEFAULTS.items()}
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "project_config.json"
    if path and not p.exists():
        # An explicit --project-config that doesn't exist must be an error:
        # silently proceeding on the built-in defaults would run the WRONG
        # recipe (demo dataset, demo step count) and only reveal it hours
        # and dollars later, inside a SageMaker job.
        raise SystemExit(f"project config not found: {p} — check the "
                         f"--project-config path")
    if p.exists():
        user = json.load(open(p))
        for key, val in user.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict) and isinstance(proj.get(key), dict):
                for k2, v2 in val.items():
                    if isinstance(v2, dict) and isinstance(proj[key].get(k2), dict):
                        proj[key][k2].update(v2)
                    else:
                        proj[key][k2] = v2
            else:
                proj[key] = val
    return proj
