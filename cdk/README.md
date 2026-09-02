# CDK: DreamZero pipeline infrastructure

Provisions, in **your own AWS account**, the durable infrastructure the
fine-tuning pipeline needs. Deploy once; then follow the root README's Quick
start to fine-tune on your dataset.

## What it creates

| Resource | Purpose |
|---|---|
| ECR repository `<project>-sagemaker-training` | holds the BYOC training/serving image |
| S3 bucket (`RETAIN` on delete) | datasets, base checkpoints, LoRA + merged models, eval results — the system of record |
| IAM role (CDK-named) | assumed by the training/merge jobs; `AmazonSageMakerFullAccess` plus explicit R/W on the bucket + ECR pull. Its ARN is a stack output + SSM parameter |
| IAM role + instance profile (CDK-named) | for the EC2 serving / Isaac Sim / eval box: SSM shell-in, ECR pull, bucket read + `sagemaker/eval-results/` write |
| CodeBuild project `<project>-image-build` | builds `docker/` and pushes the image to the ECR repo — no local Docker, no ~18GB upload over your own uplink |
| Build-kickoff custom resource (one Lambda) | starts the image build during `cdk deploy` (async — the deploy doesn't wait); re-fires only when `docker/` or `project_config.json` `image.tag` changes |
| SSM parameters `/dreamzero/<project>/*` | `bucket`, `s3_root`, `sagemaker_role_arn`, `region`, `project` (stack-written) and `image_uri` (build-written, digest-pinned) — `pipeline/pipeline_config.py` falls back to these |

Nothing is hardcoded — account and region come from your CDK environment, and
the bucket name is either provided (`-c bucket_name=…`) or CDK-generated. The
IAM role and instance-profile names are CDK-generated on purpose: IAM is a
global namespace, so fixed names would make the stack deployable only once per
account and a second region would fail at changeset validation with
`already exists`. Nothing reads those names — the pipeline uses the role **ARN**
from the stack outputs.

## The image factory

The Dockerfile compiles flash-attn from source, and the finished image is
~18GB of compressed layers in ECR (~50GB unpacked on a host), so building it in
CodeBuild (privileged `BUILD_GENERAL1_2XLARGE`, 4h timeout) next to ECR beats
building on a laptop — the push never leaves AWS. That compute is billed:
`BUILD_GENERAL1_2XLARGE` is $0.20/build-minute in us-east-1, so budget ~$12 for
the ~1h build, once per region. The `docker/` directory is
uploaded as an S3 asset at deploy time, so a build always sees the deployed
revision's Dockerfile.

**`cdk deploy` kicks the build off, but does not wait for it.** A one-call
custom resource fires `codebuild:StartBuild` during the deploy, then the
deploy returns while the ~1h build runs — blocking on it would hit
CloudFormation's custom-resource timeout, and a build failure would roll the
whole stack back. The kickoff re-fires only when the `docker/` asset content
or `project_config.json`'s `image.tag` changes; an unchanged re-deploy starts
nothing. A build can always be started by hand too (e.g. for a new tag
without a deploy):

```bash
aws codebuild start-build --project-name dreamzero-image-build \
    --environment-variables-override name=IMAGE_TAG,value=v11,type=PLAINTEXT
```

The build layer-caches from the most recently pushed tag (override with
`CACHE_FROM_TAG`), pushes `:vN`, and — only on success — writes the
digest-pinned image URI to SSM `/dreamzero/<project>/image_uri`, where
`generate_pipeline_config.py` and the pipeline's config loader pick it up.
Digest pinning matters because the repo's tags are MUTABLE: a rebuild during a
multi-day training run must never change the code an already-submitted job
runs. Its IAM role can push only to this stack's repo and pull only from it
and the public AWS Deep Learning Containers registry (`763104351884`, the
`FROM` image) — no other external registry is reachable.

## Deploy

```bash
cd cdk
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# set BOTH — the CDK CLI and boto3 read different variables (see below)
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cdk bootstrap                 # once per account/region
cdk deploy                    # add -c bucket_name=my-bucket to name the bucket
```

**Set the region explicitly, in both variables.** The stack is
environment-agnostic, so its region comes from the ambient credential chain —
and the two halves of this repo read that chain differently:

| Reader | Region resolution |
|---|---|
| CDK CLI (`cdk deploy`) | `AWS_REGION`, else `AWS_DEFAULT_REGION`, else your profile |
| boto3 (`generate_pipeline_config.py`, the pipeline) | `AWS_DEFAULT_REGION`, else your profile — **never** `AWS_REGION` |

So exporting only `AWS_REGION` deploys the stack to one region and points the
pipeline at another, and both commands succeed, so nothing tells you. Exporting
both to the same value is the only setting that cannot diverge. GPU quota is
per-region, so this matters. `cdk deploy -c region=<r>` pins the stack's own
environment but not the scripts; `--region` on `generate_pipeline_config.py`
(or `./setup.sh --region <r>`) pins the scripts but not the stack.

Deploying into more than one region of the same account works, with one caveat:
both stacks are named `dreamzero-pipeline-infra` and share the SSM prefix
`/dreamzero/<project>/*`, so whichever region is ambient when you run
`setup.sh` / `generate_pipeline_config.py` is the one that ends up in
`pipeline/pipeline_config.json`. `setup.sh` passes any `--stack` / `--region` /
`--profile` straight through to the generator, and both honour
`DREAMZERO_PIPELINE_CONFIG=<path>`, so the clean pattern is one config file per
region:

```bash
DREAMZERO_PIPELINE_CONFIG=pipeline/config.us-east-1.json ./setup.sh --region us-east-1
```

Or give the second deployment its own `-c project=<name>` — remembering that
`project` also renames the stack, the ECR repo, the CodeBuild project and the
SSM prefix, and must then be passed to every `cdk` invocation and exported as
`DREAMZERO_PROJECT` for the pipeline.

Then one command finishes the local side — it waits for the image build the
deploy kicked off (progress shown), writes `pipeline/pipeline_config.json`
from the stack outputs (digest-pinned image URI from SSM), installs the
pipeline's python deps, clones DreamZero at the validated pin, and stages the
base weights (one-time, ~128GB):

```bash
cd .. && ./setup.sh --stage-assets

# then run the pipeline on your dataset
python3 pipeline/run_pipeline.py \
    --dataset /path/to/lerobot_dataset \
    --name myrobot-v1 \
    --config pipeline/configs/aloha_bimanual_14dim.yaml
```

The manual pieces, if you want them separately:

```bash
# watch the build the deploy started
aws codebuild batch-get-builds --query 'builds[0].{phase:currentPhase,status:buildStatus}' \
    --ids $(aws codebuild list-builds-for-project \
                --project-name dreamzero-image-build --query 'ids[0]' --output text)

# write pipeline/pipeline_config.json yourself
#   (picks the digest-pinned image URI from SSM; --image-tag overrides)
python3 generate_pipeline_config.py --stack dreamzero-pipeline-infra

# local-Docker alternative to the CodeBuild factory — needs the generated
# pipeline/pipeline_config.json first:
bash ../docker/build_and_push.sh v11
```

## How the config reaches the pipeline

`generate_pipeline_config.py` reads the CloudFormation stack outputs and writes
`pipeline/pipeline_config.json` (git-ignored — it carries account-specific
values). `pipeline/pipeline_config.py` loads it, with `DREAMZERO_*` environment
variables as an override/alternative (handy for CI), and falls back to the
stack's SSM parameters (`/dreamzero/<project>/*`) for anything still missing —
so the file itself is optional once the stack is deployed and an image build
has published `image_uri`. If you deployed with a non-default
`-c project=<name>`, tell the loader which parameter path to use by setting
`DREAMZERO_PROJECT=<name>` (or keep a config file with `"project"` in it).
See `pipeline/pipeline_config.example.json` for the shape.

## Notes

- **Bucket is `RETAIN`**: `cdk destroy` leaves the bucket (and your ~500GB of
  weights + checkpoints + trained models) intact. Delete it manually only when you mean to.
  Because the bucket is CDK-named, a later re-deploy creates a *new* one and
  orphans the old — move or delete the data deliberately.
- **The retained ECR repo blocks a re-deploy into the same region.** `cdk
  destroy` also retains `<project>-sagemaker-training`, and unlike the IAM names
  that one is fixed, so the next `cdk deploy` in that region fails with
  `AWS::ECR::Repository … already exists`. Either delete the repo
  (`aws ecr delete-repository --repository-name <project>-sagemaker-training
  --force --region <r>`) or deploy with a different `-c project=<name>`. The
  repo is regional, so this never affects a *different* region.
- **Cross-region / account portability**: the whole point. Deploy the stack
  wherever you have g7e capacity; the pipeline reads the region and bucket from
  the generated config. Base weights are a one-time cross-region S3 copy.
- **Validated**: deployed into a clean account end to end — stack deploy,
  config generation, image build/push, and job submission all exercised. On a
  first deploy it is still good practice to review `cdk diff` before approving.
- **cdk-nag is wired in** — `app.py` applies the `AwsSolutions` rule pack as an
  Aspect, so **every `cdk synth` is a compliance scan** and an unsuppressed
  finding fails it. You do not need to add `AwsSolutionsChecks` yourself. The 28
  deliberate deviations are suppressed on the exact resource that carries them
  in `dreamzero_pipeline/pipeline_stack.py`, each with its justification, so a
  resource added later inherits no exemption. See
  [docs/SECURITY.md](../docs/SECURITY.md#infrastructure-findings-cdk-nag-awssolutions).
- **The stack `Description` is generated**, from the `solution` block of
  `project_config.json` via `pipeline/solution.py` (see the root
  README's *Anonymous usage metrics*) — don't hardcode it in `app.py`. Synthesis
  fails rather than emitting a description missing the id. This app deploys one
  stack, which therefore carries the bare id; if you add a second, pass
  `id_suffix=` to `stack_description()` for it, or one install gets counted
  twice. No CDK Aspect injects the SDK user-agent variable: the only
  SDK-calling compute the stack creates is the image-build kickoff Lambda (a
  custom resource making a single `StartBuild` call at deploy time, not worth
  attributing) — the consumer that matters is the merge job, which
  `pipeline/run_pipeline.py` creates at runtime and wires up directly.
