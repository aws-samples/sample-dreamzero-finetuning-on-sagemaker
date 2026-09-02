"""Durable infrastructure for the DreamZero SageMaker fine-tuning pipeline.

One stack, all a fresh account needs before running pipeline/run_pipeline.py:

  - ECR repo         : holds the BYOC training/serving image
  - S3 bucket        : datasets, checkpoints, models, eval-results (the prefix
                       layout the pipeline reads/writes; S3 is the system of record)
  - SageMaker role   : execution role the training/merge jobs assume
  - EC2 profile      : instance profile for the serving + Isaac Sim / eval box
  - CodeBuild project: builds docker/ and pushes the image to the ECR repo —
                       no local Docker, no 18GB upload from a laptop
  - SSM parameters   : /dreamzero/<project>/* — bucket, s3_root, role, region,
                       project (written here) and image_uri (written by the
                       CodeBuild build on success, digest-pinned)

Everything is parameterized (project name, optional explicit bucket name). The
stack emits CfnOutputs that `generate_pipeline_config.py` turns into the JSON
the pipeline scripts consume, and mirrors them into SSM Parameter Store so
`pipeline/pipeline_config.py` can also resolve them directly — so no account
IDs or bucket names are hardcoded anywhere in the runtime code.

`cdk deploy` also KICKS OFF the image build (a one-call custom resource fires
codebuild:StartBuild), but deliberately does not wait for it: the ~1h build
(4h cap) runs in the background while the deploy returns — a deploy that
blocked on it would hit CloudFormation's custom-resource timeout and roll the
whole stack back on a build failure. The kickoff re-fires only when the
docker/ asset content or the configured image tag (project_config.json
image.tag) changes. Builds can still be started by hand — see cdk/README.md:

  aws codebuild start-build --project-name <project>-image-build \
      --environment-variables-override name=IMAGE_TAG,value=v11,type=PLAINTEXT
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_codebuild as codebuild,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    aws_ssm as ssm,
    custom_resources as cr,
)
from cdk_nag import NagSuppressions
from constructs import Construct

# The bucket layout under s3://<bucket>/sagemaker/ (the pipeline's s3_root).
# Prefixes are created implicitly by first write; documented here for
# reference. The one prefix-scoped grant below (the serving role's write
# access) targets eval-results/ inside this layout.
S3_PREFIXES = [
    "datasets/",          # converted GEAR/yam datasets
    "checkpoints/",       # frozen base weights (Wan, tokenizer, AgiBot)
    "checkpoints-sync/",  # live training-state sync (entrypoint sync loop)
    "lora-checkpoints/",  # raw LoRA output (archival)
    "models/",            # servable merged checkpoints
    "output/",            # per-job model.tar.gz
    "eval-assets/",       # eval runner scripts (run_eval_in_job.sh + open_loop_eval.py)
    "eval-results/",      # open-loop eval outputs
]

# AWS's public Deep Learning Containers registry — the base image of
# docker/Dockerfile. The only external ECR account the image build may pull
# from; everything else resolves from the project's own repo.
DLC_ACCOUNT = "763104351884"

# Fallback tag when project_config.json carries no image.tag (app.py resolves
# it and passes image_tag=). Keep in step with docker/build_and_push.sh.
DEFAULT_IMAGE_TAG = "v11"


class DreamZeroPipelineStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, project: str,
                 bucket_name: str | None = None,
                 image_tag: str = DEFAULT_IMAGE_TAG, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)

        # --- ECR: the BYOC image ---
        # Scanner notes (checkov): tags stay MUTABLE so an image build (the
        # CodeBuild factory below, or `build_and_push.sh v11` locally) can be
        # re-run while iterating on the image; encryption is the ECR
        # default (AES-256) — switch to KMS if your org requires CMKs.
        repo = ecr.Repository(
            self, "TrainingImageRepo",
            repository_name=f"{project}-sagemaker-training",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            empty_on_delete=False,  # never nuke images on stack delete
            lifecycle_rules=[ecr.LifecycleRule(
                description="Keep the 10 most recent images",
                max_image_count=10,
            )],
        )

        # --- S3: the system of record ---
        # Bucket holds ~500GB after one full pipeline pass at the shipped
        # defaults (base weights + datasets + checkpoints + merged models —
        # see the Costs section of the root README). RETAIN on delete: these
        # artifacts must outlive any single stack (S3-first policy).
        bucket = s3.Bucket(
            self, "AssetsBucket",
            bucket_name=bucket_name,  # None → CDK auto-names uniquely
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            # versioning off deliberately: the bucket holds hundreds of GB of
            # write-once model shards; versioning every retrain would balloon
            # cost with no rollback value. Enable server access logging /
            # versioning here if your compliance baseline requires them.
            versioned=False,
            # The only way to bound orphaned multipart uploads. Every checkpoint
            # shard is an 85 GiB multipart, and a job that is killed mid-upload
            # leaves its parts behind: SIGKILL cannot be caught, so no amount of
            # care in the entrypoint can abort them. Orphaned parts are BILLED as
            # storage and appear in neither `aws s3 ls` nor ListObjectsV2, so they
            # accumulate invisibly to any accounting you or the pipeline does.
            lifecycle_rules=[s3.LifecycleRule(
                id="AbortOrphanedMultipartUploads",
                abort_incomplete_multipart_upload_after=Duration.days(7),
            )],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # --- SageMaker execution role ---
        # Training/merge jobs assume this. AmazonSageMakerFullAccess only grants
        # S3 to buckets with "sagemaker" in the name, so scope our bucket explicitly.
        #
        # The role name is deliberately CDK-generated, NOT fixed. IAM is a global
        # namespace with no region dimension, so a fixed name makes this stack
        # deployable exactly once per account: a second deploy — most likely into
        # another region because that is where your GPU quota is — fails at
        # changeset validation with "Resource of type 'AWS::IAM::Role' with
        # identifier '…' already exists". Nothing consumes the name: the ARN is
        # published as a stack output and an SSM parameter below, and that is what
        # the pipeline reads. Do not add role_name back.
        sm_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
        bucket.grant_read_write(sm_role)
        repo.grant_pull(sm_role)

        # --- EC2 instance profile for serving / eval ---
        # SSM shell-in (no SSH), pull the image from ECR, read models + write
        # eval-results. Read-only on the bucket except the eval-results prefix.
        # Name CDK-generated, for the same reason as the SageMaker role above.
        ec2_role = iam.Role(
            self, "ServingInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
            ],
        )
        bucket.grant_read(ec2_role)
        # The whole layout lives under the sagemaker/ root (s3_root), so the
        # write grant must include it — a bare "eval-results/*" pattern would
        # never match anything the eval scripts actually write.
        bucket.grant_write(ec2_role, objects_key_pattern="sagemaker/eval-results/*")
        # Instance profiles share the same global IAM namespace as roles, so this
        # name is CDK-generated too. It is the L2 construct rather than
        # CfnInstanceProfile because only the L2 exposes instance_profile_name as
        # a resolvable token — on the Cfn resource that attribute is None unless
        # you name it yourself, which would silently publish an empty
        # ServingInstanceProfileName output.
        ec2_profile = iam.InstanceProfile(
            self, "ServingInstanceProfile",
            role=ec2_role,
        )

        region = Stack.of(self).region
        account = Stack.of(self).account

        # --- SSM parameters: the stack outputs, machine-readable ---
        # pipeline/pipeline_config.py falls back to these when neither
        # pipeline_config.json nor DREAMZERO_* env vars provide a value, so a
        # fresh clone can run against a deployed stack with zero local config.
        # Fixed "/dreamzero" root + project name, so several projects coexist
        # and the loader only needs the project name to find everything.
        ssm_prefix = f"/dreamzero/{project}"
        for key, value in {
            "project": project,
            "bucket": bucket.bucket_name,
            "s3_root": f"s3://{bucket.bucket_name}/sagemaker",
            "sagemaker_role_arn": sm_role.role_arn,
            "region": region,
        }.items():
            ssm.StringParameter(
                self, f"Param{key.title().replace('_', '')}",
                parameter_name=f"{ssm_prefix}/{key}",
                string_value=value,
            )
        # image_uri is NOT created here: the CodeBuild project below writes it
        # (digest-pinned) after each successful build. Creating it with a
        # placeholder would make it CloudFormation-managed and turn every
        # build's PutParameter into stack drift.
        image_param_name = f"{ssm_prefix}/image_uri"

        # --- CodeBuild image factory ---
        # Builds docker/ into the ECR repo. The Dockerfile compiles flash-attn
        # from source and the image is ~18GB compressed, so building anywhere
        # but close to ECR wastes an hour of upload; BUILD_GENERAL1_2XLARGE
        # gives the cores the compile wants and enough disk for the layers.
        # The docker/ directory ships as an S3 asset — each `cdk deploy`
        # re-uploads it when its content changed, so a build always sees the
        # docker/ tree of the deployed revision.
        docker_src = s3_assets.Asset(
            self, "DockerBuildContext",
            path=str(Path(__file__).resolve().parent.parent.parent / "docker"),
        )
        build = codebuild.Project(
            self, "ImageFactory",
            project_name=f"{project}-image-build",
            description=f"Builds docker/ and pushes :<IMAGE_TAG> to the "
                        f"{project}-sagemaker-training ECR repo; on success "
                        f"writes the digest-pinned URI to SSM {image_param_name}",
            source=codebuild.Source.s3(
                bucket=docker_src.bucket, path=docker_src.s3_object_key),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.X2_LARGE,
                privileged=True,  # docker-in-docker: required to build images
            ),
            environment_variables={
                "ECR_REPO_URI": codebuild.BuildEnvironmentVariable(
                    value=repo.repository_uri),
                "ECR_REPO_NAME": codebuild.BuildEnvironmentVariable(
                    value=repo.repository_name),
                "DLC_ACCOUNT": codebuild.BuildEnvironmentVariable(
                    value=DLC_ACCOUNT),
                # from project_config.json image.tag (via app.py); override
                # per build:  aws codebuild start-build ...
                #   --environment-variables-override
                #     name=IMAGE_TAG,value=v11,type=PLAINTEXT
                "IMAGE_TAG": codebuild.BuildEnvironmentVariable(
                    value=image_tag),
                # "auto" → the buildspec caches from the most recently pushed
                # tag in the repo; set a tag explicitly to pin the cache
                # source ("auto" rather than "" — CodeBuild rejects empty
                # environment-variable values)
                "CACHE_FROM_TAG": codebuild.BuildEnvironmentVariable(value="auto"),
                "SSM_IMAGE_PARAM": codebuild.BuildEnvironmentVariable(
                    value=image_param_name),
            },
            timeout=Duration.hours(4),  # flash-attn compile dominates
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    # Log in to both registries here; docker keeps the auth in
                    # its config file, which persists across phases.
                    "pre_build": {"commands": [
                        'aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | '
                        'docker login --username AWS --password-stdin "${ECR_REPO_URI%%/*}"',
                        'aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | '
                        'docker login --username AWS --password-stdin '
                        '"${DLC_ACCOUNT}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"',
                    ]},
                    # One shell block: cache detection, pull and build share
                    # variables without relying on cross-phase env persistence.
                    "build": {"commands": ["\n".join([
                        'set -e',
                        'if [ "$CACHE_FROM_TAG" = "auto" ]; then',
                        '  CACHE_FROM_TAG=$(aws ecr describe-images'
                        ' --repository-name "$ECR_REPO_NAME"'
                        ' --query "sort_by(imageDetails[?imageTags],&imagePushedAt)[-1].imageTags[0]"'
                        ' --output text 2>/dev/null || true)',
                        'fi',
                        # No docker pull of the cache image: BuildKit reads
                        # --cache-from inline-cache metadata straight from the
                        # registry, so pulling ~18GB first is pure waste (and
                        # an image built without BUILDKIT_INLINE_CACHE yields
                        # no hits either way — the first factory build after
                        # such an image is simply uncached).
                        'CACHE_ARGS=""',
                        'if [ -n "$CACHE_FROM_TAG" ] && [ "$CACHE_FROM_TAG" != "None" ]; then',
                        '  echo "cache source: $ECR_REPO_URI:$CACHE_FROM_TAG"',
                        '  CACHE_ARGS="--cache-from $ECR_REPO_URI:$CACHE_FROM_TAG"',
                        'fi',
                        # BUILDKIT_INLINE_CACHE embeds cache metadata in the
                        # pushed image so the NEXT build can --cache-from it.
                        'DOCKER_BUILDKIT=1 docker build'
                        ' --build-arg BUILDKIT_INLINE_CACHE=1'
                        ' --build-arg DLC_REGION="$AWS_DEFAULT_REGION"'
                        ' $CACHE_ARGS -t "$ECR_REPO_URI:$IMAGE_TAG" .',
                    ])]},
                    # post_build runs even after a failed build phase — gate
                    # on CODEBUILD_BUILD_SUCCEEDING so a broken image is never
                    # pushed and the SSM pointer never moves to one.
                    "post_build": {"commands": ["\n".join([
                        'if [ "$CODEBUILD_BUILD_SUCCEEDING" != "1" ]; then',
                        '  echo "build failed — not pushing, not touching SSM"; exit 1',
                        'fi',
                        'set -e',
                        'docker push "$ECR_REPO_URI:$IMAGE_TAG"',
                        'DIGEST=$(aws ecr describe-images'
                        ' --repository-name "$ECR_REPO_NAME"'
                        ' --image-ids imageTag="$IMAGE_TAG"'
                        ' --query "imageDetails[0].imageDigest" --output text)',
                        # digest-pinned: the repo has MUTABLE tags, so a tag
                        # reference could silently change under a running
                        # multi-day job; the digest cannot.
                        'aws ssm put-parameter --name "$SSM_IMAGE_PARAM"'
                        ' --type String --overwrite'
                        ' --value "$ECR_REPO_URI@$DIGEST"',
                        'echo "pushed $ECR_REPO_URI:$IMAGE_TAG ($DIGEST)"',
                        'echo "SSM $SSM_IMAGE_PARAM -> $ECR_REPO_URI@$DIGEST"',
                    ])]},
                },
            }),
        )
        docker_src.grant_read(build)
        repo.grant_pull_push(build)
        # grant_pull_push covers layer/image push-pull but NOT DescribeImages,
        # which the buildspec needs twice: auto-detecting the cache tag and
        # resolving the pushed digest for the SSM write.
        build.add_to_role_policy(iam.PolicyStatement(
            sid="DescribeImagesForCacheAndDigest",
            actions=["ecr:DescribeImages"],
            resources=[repo.repository_arn],
        ))
        # Pull-only on the public DLC registry (the FROM image) — the single
        # allowlisted external ECR account; no other registry is reachable.
        build.add_to_role_policy(iam.PolicyStatement(
            sid="DlcBaseImagePull",
            actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
                     "ecr:BatchCheckLayerAvailability"],
            resources=[f"arn:{self.partition}:ecr:{region}:{DLC_ACCOUNT}:repository/*"],
        ))
        build.add_to_role_policy(iam.PolicyStatement(
            sid="PublishImageUriParam",
            actions=["ssm:PutParameter"],
            resources=[f"arn:{self.partition}:ssm:{region}:{account}:"
                       f"parameter{image_param_name}"],
        ))

        # --- Deploy-time build kickoff ---
        # One StartBuild call during `cdk deploy`, asynchronous by design: the
        # deploy returns while the ~1h build runs (blocking on it would hit
        # the custom-resource timeout and roll the stack back on any build
        # failure). Re-fires only when this resource's properties change —
        # i.e. when the docker/ asset content or the configured tag changes;
        # an unchanged re-deploy starts nothing. DOCKER_SRC_HASH doubles as
        # that change detector and as provenance in the build's environment.
        kickoff_call = cr.AwsSdkCall(
            service="CodeBuild",
            action="startBuild",
            parameters={
                "projectName": build.project_name,
                "environmentVariablesOverride": [
                    {"name": "IMAGE_TAG", "value": image_tag,
                     "type": "PLAINTEXT"},
                    {"name": "DOCKER_SRC_HASH", "value": docker_src.asset_hash,
                     "type": "PLAINTEXT"},
                ],
            },
            physical_resource_id=cr.PhysicalResourceId.of(
                f"{project}-image-build-kickoff"),
            # startBuild echoes the whole build object; CloudFormation caps
            # custom-resource data at 4KB, so keep only the build id
            output_paths=["build.id"],
        )
        kickoff = cr.AwsCustomResource(
            self, "ImageBuildKickoff",
            resource_type="Custom::ImageBuildKickoff",
            on_create=kickoff_call,
            on_update=kickoff_call,
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(actions=["codebuild:StartBuild"],
                                    resources=[build.project_arn]),
            ]),
            install_latest_aws_sdk=False,  # the bundled SDK covers StartBuild
        )
        kickoff.node.add_dependency(build)

        # --- outputs the pipeline scripts consume ---
        CfnOutput(self, "Project", value=project, export_name=f"{cid}-project")
        CfnOutput(self, "BucketName", value=bucket.bucket_name,
                  export_name=f"{cid}-bucket")
        CfnOutput(self, "S3Root", value=f"s3://{bucket.bucket_name}/sagemaker",
                  export_name=f"{cid}-s3root")
        CfnOutput(self, "EcrRepoUri", value=repo.repository_uri,
                  export_name=f"{cid}-ecr")
        CfnOutput(self, "SageMakerRoleArn", value=sm_role.role_arn,
                  export_name=f"{cid}-smrole")
        CfnOutput(self, "ServingInstanceProfileName",
                  value=ec2_profile.instance_profile_name,
                  export_name=f"{cid}-ec2profile")
        CfnOutput(self, "Region", value=region)
        CfnOutput(self, "Account", value=account)
        CfnOutput(self, "ImageBuildProject", value=build.project_name,
                  export_name=f"{cid}-imagebuild")
        CfnOutput(self, "SsmParameterPrefix", value=ssm_prefix)

        # --- cdk-nag suppressions ---
        # app.py runs the AwsSolutions rule pack over every synth. Each entry
        # below is a deliberate, documented deviation — the reason carries the
        # evidence, and the suppression sits on the exact resource so a new
        # resource added later is still linted with no exemption.
        NagSuppressions.add_resource_suppressions(bucket, [{
            "id": "AwsSolutions-S1",
            "reason": "Server access logging is deliberately off: the bucket "
                      "holds write-once model shards and datasets accessed "
                      "only by this pipeline's jobs, and logging every "
                      "multi-GB training read would add cost without audit "
                      "value here. Enable it (and versioning) if your "
                      "compliance baseline requires them — see the bucket "
                      "definition comment.",
        }])
        NagSuppressions.add_resource_suppressions(sm_role, [
            {
                "id": "AwsSolutions-IAM4",
                "reason": "AmazonSageMakerFullAccess is the documented "
                          "baseline for SageMaker training execution roles; "
                          "its own S3 access is limited to buckets with "
                          "'sagemaker' in the name, and this stack's bucket "
                          "is granted explicitly below it.",
                "applies_to": ["Policy::arn:<AWS::Partition>:iam::aws:policy/"
                               "AmazonSageMakerFullAccess"],
            },
            {
                "id": "AwsSolutions-IAM5",
                "reason": "Wildcards come from two CDK grants: "
                          "bucket.grant_read_write (object-level * scoped to "
                          "the single assets-bucket ARN — training jobs "
                          "read/write run-named prefixes created at runtime) "
                          "and repo.grant_pull (ecr:GetAuthorizationToken "
                          "supports only Resource:*).",
            },
        ], apply_to_children=True)
        NagSuppressions.add_resource_suppressions(ec2_role, [
            {
                "id": "AwsSolutions-IAM4",
                "reason": "AmazonSSMManagedInstanceCore is the standard way "
                          "to shell into the serving/eval box without SSH or "
                          "inbound ports; AmazonEC2ContainerRegistryReadOnly "
                          "is pull-only for the serving image.",
                "applies_to": [
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/"
                    "AmazonSSMManagedInstanceCore",
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/"
                    "AmazonEC2ContainerRegistryReadOnly",
                ],
            },
            {
                "id": "AwsSolutions-IAM5",
                "reason": "Wildcards come from bucket.grant_read on the "
                          "single assets bucket and bucket.grant_write "
                          "restricted to the sagemaker/eval-results/* prefix "
                          "— the narrowest grants the L2 constructs emit.",
            },
        ], apply_to_children=True)
        NagSuppressions.add_resource_suppressions(build, [
            {
                "id": "AwsSolutions-CB4",
                "reason": "The project produces no S3 build artifacts to "
                          "encrypt with a CMK: its output is a docker image "
                          "pushed to ECR, encrypted at rest with the ECR "
                          "default (AES-256). Switch the repo to a KMS key "
                          "if your org requires CMKs (see the ECR repo "
                          "comment).",
            },
            {
                "id": "AwsSolutions-IAM5",
                "reason": "Wildcards come from repo.grant_pull_push on the "
                          "project's own ECR repo, the docker/ asset "
                          "bucket grant, the pull-only statement on AWS's "
                          "public DLC registry account (repository names "
                          "there are not enumerable ahead of time), and the "
                          "CDK-generated CloudWatch Logs/CodeBuild-reports "
                          "grants scoped to this project's own ARNs.",
            },
        ], apply_to_children=True)
        # The AwsCustomResource kickoff installs a CDK-managed singleton
        # Lambda under a stable hashed id; its service role carries only
        # AWSLambdaBasicExecutionRole (CloudWatch Logs write).
        cr_handler = self.node.try_find_child(
            "AWS679f53fac002430cb0da5b7982bd2287")
        if cr_handler is not None:
            NagSuppressions.add_resource_suppressions(cr_handler, [{
                "id": "AwsSolutions-IAM4",
                "reason": "CDK-managed AwsCustomResource handler; "
                          "AWSLambdaBasicExecutionRole grants only CloudWatch "
                          "Logs write for its own log group.",
                "applies_to": ["Policy::arn:<AWS::Partition>:iam::aws:policy/"
                               "service-role/AWSLambdaBasicExecutionRole"],
            }, {
                "id": "AwsSolutions-L1",
                "reason": "The runtime of this handler is chosen by aws-cdk-lib, "
                          "not by this stack, so the rule is unfixable from here "
                          "and fires purely on version skew: cdk/requirements.txt "
                          "pins neither aws-cdk-lib nor cdk-nag, and on Python 3.9 "
                          "pip resolves aws-cdk-lib to its last 3.9 release while "
                          "still taking the newest cdk-nag, which knows a newer "
                          "Node family than that CDK emits. Because app.py treats "
                          "any unsuppressed finding as an error, leaving this "
                          "unsuppressed makes `cdk synth` exit 1 on the Python "
                          "floor the README documents. The handler runs once at "
                          "deploy time, in-account, to start a CodeBuild build; "
                          "it is not internet-facing and processes no input.",
            }], apply_to_children=True)
