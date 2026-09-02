#!/usr/bin/env python3
"""CDK app entrypoint for the DreamZero fine-tuning pipeline infrastructure.

Deploys one stack that provisions everything a fresh AWS account needs to run
the pipeline (pipeline/run_pipeline.py): an ECR repo for the BYOC image, an S3
bucket with the prefix layout, a SageMaker execution role, and an EC2 instance
profile for serving/evaluation.

Configure via CDK context (cdk.json or -c flags); nothing is hardcoded:
  cdk deploy \
    -c project=my-robot -c bucket_name=my-dreamzero-bucket

`project` (default "dreamzero", set in cdk.json) namespaces the stack name, the
ECR repo, the CodeBuild project and the SSM prefix — give a second deployment in
the same account its own value if you want those separated. The IAM roles are
CDK-named and never collide either way.

Account/region come from the standard CDK environment (CDK_DEFAULT_ACCOUNT /
CDK_DEFAULT_REGION, i.e. your profile, AWS_REGION or AWS_DEFAULT_REGION), or
from an explicit -c account=<id> / -c region=<region>. Export both region
variables: boto3 does not read AWS_REGION, so the runtime scripts can otherwise
resolve a different region than the one this stack deploys into — see the
"Set the region explicitly" section of cdk/README.md.
"""
import json
import sys
from pathlib import Path

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from dreamzero_pipeline.pipeline_stack import DEFAULT_IMAGE_TAG, DreamZeroPipelineStack

# The solution id/name/version live in the 'solution' block of the repo-root
# project_config.json, shared with the runtime scripts (which read it
# through pipeline/solution.py), so the stack Description and the SDK
# user-agent suffix can never drift apart across a release. Same cross-directory
# import idiom the runtime scripts use (see evaluation/submit_eval_job.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from solution import stack_description  # noqa: E402

app = cdk.App()
# Cost-cleanup exemption tags for ALL CloudFormation-managed infra in this
# stack. Shared and sandbox accounts often run scheduled cleanup tooling that
# stops or deletes untagged resources; these are common exemption tag names —
# swap in whatever your account's tooling honours. (SageMaker training jobs are
# created at RUNTIME, not by CDK, so they're tagged separately via
# extra_job_tags in pipeline_config.json — see pipeline/README.md Caveats.)
cdk.Tags.of(app).add("auto-delete", "no")
cdk.Tags.of(app).add("auto-stop", "no")
cdk.Tags.of(app).add("name", "prototype")

project = app.node.try_get_context("project") or "dreamzero"
bucket_name = app.node.try_get_context("bucket_name")  # optional; None → CDK auto-names

# The tag the deploy-time kickoff build pushes comes from project_config.json
# (image.tag), so retagging is a config edit — bumping it and re-deploying
# triggers a rebuild. Missing file/block degrades to the stack default.
try:
    _proj = json.loads((Path(__file__).resolve().parent.parent
                        / "project_config.json").read_text())
except (OSError, json.JSONDecodeError):
    _proj = {}
image_tag = (_proj.get("image") or {}).get("tag") or DEFAULT_IMAGE_TAG

env = cdk.Environment(
    account=app.node.try_get_context("account") or None,
    region=app.node.try_get_context("region") or None,
)

DreamZeroPipelineStack(
    app,
    f"{project}-pipeline-infra",
    project=project,
    bucket_name=bucket_name,
    image_tag=image_tag,
    env=env,
    # This app deploys exactly one stack, so the description is just the
    # solution id, name and version — no component label, no id suffix. A
    # second stack added later must pass id_suffix= — that is what keeps a
    # single install from being counted twice — and may pass component to
    # label it in the console (see solution.stack_description).
    description=stack_description(),
)

# cdk-nag: every synth is linted against the AwsSolutions rule pack and FAILS
# on unsuppressed errors. The deliberate deviations are suppressed at the
# resource that carries them (see NagSuppressions calls in pipeline_stack.py),
# each with the reason inline — never blanket-suppressed here at the app level.
cdk.Aspects.of(app).add(AwsSolutionsChecks())

app.synth()
