#!/usr/bin/env python3
"""Submit an open-loop evaluation of DreamZero checkpoints as a SageMaker job.

Compares up to 3 model arms on the SAME deterministic frame set (evenly spaced
across the dataset — no RNG, so results are comparable across runs and arms).

Usage:
  python3 submit_eval_job.py --dataset-s3 s3://<bucket>/sagemaker/datasets/mydata/ \
      --arm finetuned=s3://<bucket>/sagemaker/models/mymodel-merged/ \
      --arm base=s3://<bucket>/sagemaker/checkpoints/DreamZero-AgiBot/:base \
      --config-donor s3://<bucket>/sagemaker/models/mymodel-merged/ \
      --results-s3 s3://<bucket>/sagemaker/eval-results/myrun/

The ':base' suffix marks an arm as a never-fine-tuned base checkpoint: its
weight shards are composed with --config-donor's config/experiment_cfg (which
carry your embodiment's transforms/stats) — the same composition training uses.

MSE interpretation caveat: open-loop MSE measures imitation on-distribution.
If your action labels are close to current state (small lead), a state-echoing
model scores deceptively well — treat closed-loop behavior as the real test.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from pipeline_config import (boto_session, job_tags, load_config,  # noqa: E402
                             load_project_config)
from solution import get_client  # noqa: E402


def channel(name, uri):
    return {"ChannelName": name, "DataSource": {"S3DataSource": {
        "S3DataType": "S3Prefix", "S3Uri": uri,
        "S3DataDistributionType": "FullyReplicated"}}, "InputMode": "File"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-s3", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help="name=s3://prefix[/][:base]  (repeat, max 3)")
    ap.add_argument("--config-donor", default=None,
                    help="s3 prefix of a fine-tuned checkpoint whose config/"
                         "experiment_cfg is composed over ':base' arms")
    ap.add_argument("--results-s3", required=True)
    ap.add_argument("--num-samples", type=int, default=150)
    ap.add_argument("--embodiment-tag", default="yam",
                    help="EmbodimentTag the checkpoints were trained with "
                         "(yam for GEAR bimanual, oxe_droid for DROID)")
    ap.add_argument("--instance-type", default=None,
                    help="default: project_config.json compute.eval.instance_type")
    ap.add_argument("--project-config", default=None,
                    help="per-run project_config.json (same file run_pipeline.py "
                         "takes); its compute.eval block sets the instance type. "
                         "Default: the repo root's project_config.json")
    ap.add_argument("--eval-script-s3", default=None,
                    help="s3 prefix holding open_loop_eval.py + run_eval_in_job.sh "
                         "(default: <s3_root>/eval-assets/)")
    args = ap.parse_args()

    cfg = load_config()
    eval_compute = load_project_config(args.project_config)["compute"]["eval"]
    instance_type = args.instance_type or eval_compute["instance_type"]
    evalassets = args.eval_script_s3 or f"{cfg['s3_root']}/eval-assets/"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"dreamzero-eval-{stamp}"

    chans = [channel("wan", f"{cfg['s3_root']}/checkpoints/Wan2.1-I2V-14B-480P/"),
             channel("tokenizer", f"{cfg['s3_root']}/checkpoints/umt5-xxl-tokenizer/"),
             channel("dataset", args.dataset_s3),
             channel("evalscript", evalassets)]
    if len(args.arm) > 3:
        sys.exit(f"ERROR: {len(args.arm)} arms given; max 3 (each arm mounts a "
                 "~90GB checkpoint channel)")
    arms_env, vol, seen_names = [], 120, set()
    for i, spec in enumerate(args.arm):
        name, eq, rest = spec.partition("=")
        # ':base' is a suffix AFTER the s3 uri — partition(':') would split
        # inside 's3://', sending S3Uri='s3' to SageMaker
        if rest.endswith(":base"):
            uri, kind = rest[:-len(":base")], "base"
        else:
            uri, kind = rest, ""
        if not eq or not uri.startswith("s3://"):
            sys.exit(f"ERROR: bad --arm {spec!r}; expected name=s3://prefix[/][:base]")
        # the name becomes an S3 result prefix and a ':'-delimited shell field
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            sys.exit(f"ERROR: arm name {name!r} must match [A-Za-z0-9_-]+ "
                     "(it is used as an S3 prefix and a shell field)")
        if name in seen_names:
            sys.exit(f"ERROR: duplicate arm name {name!r} — the second arm's "
                     "results would overwrite the first's in S3")
        seen_names.add(name)
        ch_name = f"model{i}"
        chans.append(channel(ch_name, uri if uri.endswith("/") else uri + "/"))
        arms_env.append(f"{name}:/opt/ml/input/data/{ch_name}" + (":base" if kind == "base" else ""))
        vol += 100
    env = {"RESULTS_S3_URI": args.results_s3.rstrip("/"),
           "NUM_SAMPLES": str(args.num_samples),
           "EMBODIMENT_TAG": args.embodiment_tag,
           "ARMS": " ".join(arms_env)}
    if args.config_donor:
        chans.append(channel("donor", args.config_donor if args.config_donor.endswith("/")
                             else args.config_donor + "/"))
        env["BASE_CONFIG_DONOR"] = "/opt/ml/input/data/donor"
        vol += 100  # File mode downloads the donor's full ~92GB checkpoint too

    sm = get_client("sagemaker", session=boto_session(cfg))
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={
            "TrainingImage": cfg["image_uri"], "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "/opt/ml/input/data/evalscript/run_eval_in_job.sh"]},
        RoleArn=cfg["sagemaker_role_arn"],
        InputDataConfig=chans,
        OutputDataConfig={"S3OutputPath": f"{cfg['s3_root']}/output/"},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1,
                        "VolumeSizeInGB": vol + 100},
        StoppingCondition={
            "MaxRuntimeInSeconds": int(float(eval_compute["max_runtime_hours"]) * 3600)},
        Environment=env,
        Tags=job_tags(cfg, "eval"))
    print(f"submitted: {job}")
    print(f"results will land at: {args.results_s3.rstrip('/')}/<arm>/mse.txt")


if __name__ == "__main__":
    main()
