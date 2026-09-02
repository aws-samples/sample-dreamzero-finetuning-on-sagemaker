# Security deep dive

← back to the [README](../README.md#security)

This page expands the README's security summary. It splits the material into
what was **found and resolved in code**, what was **found and accepted with a
written reason**, and what is **deliberately left for you to fix** before this
goes anywhere near production.

Nothing here is aspirational: every row cites the file and line that carries
the fix or the suppression, and every count is reproducible with the commands
in the next section.

## Scan coverage

| Scanner | How it runs | Scope | Result |
|---|---|---|---|
| [cdk-nag](https://github.com/cdklabs/cdk-nag) `AwsSolutions` | wired into `cdk/app.py` — **every** `cdk synth` | the synthesized CloudFormation template | 41 rule evaluations across 13 resource paths (7 constructs), **0 unsuppressed** — an unsuppressed finding **fails the synth**. The Compliant/Suppressed split moves with the resolved `aws-cdk-lib`, because one rule (`AwsSolutions-L1`, on CDK's own `AwsCustomResource` handler) depends on the Node runtime that CDK emits: 13/28 on Python 3.10+ (aws-cdk-lib 2.267.0), 12/29 on the documented 3.9 floor (2.255.0, where the rule fires and `pipeline_stack.py:486-498` suppresses it). Read your own numbers from `cdk.out/AwsSolutions--dreamzero-pipeline-infra-NagReport.csv` |
| Bandit 1.8.6 | `bandit -r pipeline evaluation inference docker cdk/app.py cdk/dreamzero_pipeline cdk/generate_pipeline_config.py` | 3,855 lines of Python | **0 High, 0 Medium.** 12 Low in shipped code, 66 more in `pipeline/tests/` (55 of them `assert`) |
| Checkov | Dockerfile / IaC rules | `docker/Dockerfile` | 2 findings, both skipped inline with a reason |
| `pip-audit` (PyPI + OSV) | `docker run` against the built image — see [DEPENDENCY-INVENTORY.md](DEPENDENCY-INVENTORY.md) | the 382 Python packages **installed in image `v11`**, not the requirements files | 7 packages, **33 unique advisories**; 1 assessed reachable (PYSEC-2026-2288, same precondition as TB6) |
| Semgrep | Dockerfile audit rules | `docker/Dockerfile` | 1 finding, `nosemgrep` inline with a reason |
| ECR image scanning | `image_scan_on_push=True` (`pipeline_stack.py:92`) | every image the factory pushes | continuous; findings appear in the ECR console — see "Known gaps" #3, nothing acts on them for you |

Reproduce the Python scan yourself:

```bash
pip install bandit
bandit -r pipeline evaluation inference docker \
       cdk/app.py cdk/dreamzero_pipeline cdk/generate_pipeline_config.py
```

Reproduce the infrastructure scan — this is just a normal synth, because the
rule pack is part of the app:

```bash
cd cdk && cdk synth      # exits non-zero if any new finding is unsuppressed
```

## Code-level hardening

Every High- and Medium-severity static-analysis finding was **fixed in code**,
not silenced. Where a finding is a false positive for this workload, the
suppression sits on the exact line with its justification, so an auditor never
has to guess.

| Finding | File(s) | Resolution |
|---|---|---|
| **B202 `tarfile.extractall`** (High) — a crafted `model.tar.gz` could write outside the extraction directory | `pipeline/run_pipeline.py:460-473` | **Fixed.** Extraction uses `filter="data"` (rejects `..`, absolute paths, links, devices, setuid bits) on Python 3.12+, with an explicit per-member validation loop as the fallback on older interpreters. `# nosec B202` carries the reasoning for *both* paths inline. |
| **B615 unpinned `snapshot_download`** (Medium ×2) — a hijacked upstream `main` would silently change the base weights you train against | `pipeline/stage_base_assets.py:139-140`, `pipeline/run_pipeline.py:515-517` | **Fixed.** All three base repos are pinned to commit SHAs in `stage_base_assets.py::JOBS`; the dataset download passes the `revision` from config (the shipped default is the `v2.1` tag). `# nosec B615` records that a revision is always supplied. |
| **CKV_DOCKER_3** — image runs as `root` | `docker/Dockerfile:16` | **Skipped with reason.** SageMaker mounts `/opt/ml/{input,model,checkpoints}` root-owned and its checkpoint-sync agent requires root; the image is a batch process, never a network service. Fully discussed in "Known gaps" #1 — this is the one item you may still want to change. |
| **CKV_DOCKER_2** — no `HEALTHCHECK` | `docker/Dockerfile:20` | **Skipped with reason.** `HEALTHCHECK` is a long-running-service mechanism; SageMaker training jobs are batch and the platform never consults it. |
| **Semgrep `dockerfile-pip-extra-index-url`** — extra package index widens the supply chain | `docker/Dockerfile:71` | **Skipped with reason.** The extra index is PyTorch's own CUDA 12.9 wheel host (`download.pytorch.org/whl/cu129`), the documented install path for these builds; every other package resolves from PyPI. The `nosemgrep` marker sits on the line *directly* above the `RUN`: semgrep honours it only on the same or the immediately preceding line, so a comment inserted between the two silently voids the suppression. |
| **Third-party source pulled at image-build time** | `docker/Dockerfile:26`, `setup.sh:48-84` | **Pinned.** Upstream DreamZero is cloned at commit `ab790c19…`, never a branch. `setup.sh` clones the same pin locally and, if an existing clone differs, **warns and refuses to reset it** rather than silently moving someone's tree. |
| **Mutable ECR tag could change what a queued job runs** | CodeBuild `post_build` → SSM → `pipeline_config.py` | **Fixed by design.** The image factory writes a **digest-pinned** URI to SSM and jobs reference the digest, so re-pushing a tag cannot alter a job that is already queued or running. |
| **Servable weights overwritten in place** | `pipeline/run_pipeline.py::launch_merge` | **Fixed.** The merge stage lists `models/<name>-merged/` first and aborts if it already holds objects, rather than half-overwriting the weights something may be serving. |

## Findings accepted, with reasons

The 12 remaining Bandit findings are all **Low severity** and all in the same
two families:

- **B404 / B603 (10 findings)** — `import subprocess` and the calls that use
  it, in `convert_lerobot_v3_to_v21.py`, `prep_dataset.py`,
  `stage_base_assets.py` and `run_pipeline.py`. Every call passes a **list
  argv with `shell=False`**; the executables are `aws`, `git`, `ffmpeg` and
  `python`. No user-supplied string is ever concatenated into a
  shell command line. Bandit flags the module import and the call shape, not
  an actual injection.
- **B110 (2 findings)** — `try/except/pass` in two optional-metadata readers
  (`open_loop_eval.py:169` reading a task annotation,
  `convert_lerobot_v3_to_v21.py:147` reading a cached frame count) where the
  field being absent is a normal, handled case with a documented fallback on
  the next line.

The 61 additional findings under `pipeline/tests/` are test-suite artefacts:

- **53 × B101 `assert_used`** — these are unit tests; `assert` is the point.
- **1 × B106 `hardcoded_password_funcarg`** — `test_solution.py:54` builds a
  `boto3.Session` from AWS's published example access-key id and a literal
  spelled `test-secret-not-a-real-key`. It is not a credential, and the
  session is only used to read back a user-agent string — it never calls AWS.
- **1 × B607, 4 × B603, 2 × B404** — the test that shells out to `git ls-files`
  to assert which files actually ship.

If you fork this and wire Bandit into CI, scan the shipped code and the tests
with different baselines, or you will spend your time triaging `assert`.

## Infrastructure findings (cdk-nag `AwsSolutions`)

`cdk/app.py` applies the `AwsSolutions` rule pack as a CDK Aspect, so **every
synth is a scan** and any finding without a suppression fails the build. The
findings this stack produces — 28, or 29 including the version-dependent
`AwsSolutions-L1` described in "Scan coverage" — are suppressed in
`cdk/dreamzero_pipeline/pipeline_stack.py:393-499`, each on the exact resource
that carries it:

| Rule | Count | Resource(s) | Why it is suppressed |
|---|---|---|---|
| `AwsSolutions-S1` | 1 | assets bucket | Server access logging is off deliberately: the bucket holds write-once model shards read only by this pipeline's own jobs, and logging every multi-GB training read would add cost without audit value. See "Known gaps" #2. |
| `AwsSolutions-IAM4` | 4 | SageMaker execution role, EC2 instance profile, custom-resource Lambda | AWS-managed baseline policies: `AmazonSageMakerFullAccess` (its S3 reach is limited to buckets with "sagemaker" in the name; this stack's bucket is granted explicitly on top), `AmazonSSMManagedInstanceCore`, `AmazonEC2ContainerRegistryReadOnly` (pull-only), `AWSLambdaBasicExecutionRole` (Logs write only). |
| `AwsSolutions-IAM5` | 22 | SageMaker execution role, EC2 instance profile, CodeBuild project | Wildcards emitted by the CDK L2 grants, not hand-written: `bucket.grant_read_write` (object-level `*` scoped to the one bucket ARN — jobs create run-named prefixes at runtime), `repo.grant_pull`/`grant_pull_push` (`ecr:GetAuthorizationToken` supports only `Resource:*`), `bucket.grant_write` narrowed to `sagemaker/eval-results/*`, the pull-only statement on AWS's public DLC registry account (repository names there are not enumerable in advance), and CDK-generated Logs/reports grants scoped to this project's own ARNs. |
| `AwsSolutions-CB4` | 1 | CodeBuild project | There is no S3 build artifact to encrypt with a CMK — the project's output is a docker image pushed to ECR, encrypted at rest with the ECR default (AES-256). See "Known gaps" #7. |

Two details that make these suppressions honest rather than blanket:

- Each `AwsSolutions-IAM4` entry sets **`applies_to`** with the specific
  managed-policy ARN, so attaching a *fifth* managed policy later is a fresh
  finding and fails the synth.
- The suppressions are attached to named resources with
  `apply_to_children=True`, **not** to the stack. A resource added tomorrow
  inherits no exemption.

### What the stack does provide

For balance — the controls that are on by default, none of which needed a
suppression:

- S3: `BlockPublicAccess.BLOCK_ALL`, SSE-S3 encryption at rest, `enforce_ssl`
  (denies non-TLS requests), `RemovalPolicy.RETAIN`.
- ECR: scan-on-push enabled, lifecycle rule capping stored images at 10,
  `empty_on_delete=False` so a stack delete cannot destroy your images.
- Compute access: the serving/eval instance uses **SSM Session Manager** — no
  inbound ports, no SSH key material.
- Jobs: a `StoppingCondition` with a per-stage `max_runtime_hours` on every
  submission, and a read-only preflight (bucket, image, role, quota) that runs
  before any spend.
- Image integrity: jobs reference the image **by digest**, not by tag.
- No secrets in the build: the image build's only credential is the
  short-lived ECR token CodeBuild obtains at runtime.

## Threat model summary

Trust boundaries this sample crosses:

| # | Boundary | What crosses it |
|---|---|---|
| TB1 | Your workstation ↔ your AWS account | AWS credentials from the ambient chain or a named profile |
| TB2 | AWS account ↔ Hugging Face Hub | dataset and base-weight downloads over the public internet |
| TB3 | AWS account ↔ AWS's public DLC registry (`763104351884`) | the docker base image pull |
| TB4 | Job container ↔ S3 bucket | execution-role credentials, inside the job |
| TB5 | This repo ↔ its readers | everything committed here is public |
| TB6 | S3 bucket ↔ what a job **executes** | the merge and eval jobs run a *script* fetched from S3, and the training job restores a *pickled* checkpoint from S3, so bucket write is container code execution on three prefixes — see "Known gaps" #10 |

The highest-consequence risks, and where they land:

| Risk | Status |
|---|---|
| Hijacked upstream repo changes the weights or code you train against (TB2) | **Mitigated** — base repos pinned to commit SHAs, upstream code pinned to a commit, dataset pinned to a tag |
| Path traversal out of a downloaded `model.tar.gz` (TB4 → workstation) | **Mitigated** — filtered extraction with a validating fallback |
| Wrong-account or wrong-region execution (TB1) | **Mitigated** — read-only preflight resolves bucket/image/role before any spend; `boto_session` pins the region from config so the SDK cannot follow a stray `AWS_REGION`. Residual: the `aws s3 sync` CLI calls use the ambient region, which is why the README insists on setting both region variables |
| Runaway training spend (the most likely real incident) | **Partly mitigated** — preflight quota check, a cheap smoke job as the gate, per-stage runtime caps, a measured cost table. Residual: the driver continues through train and merge without prompting. See "Known gaps" #6 |
| Silent substitution of a staged weight shard in S3 | **Accepted** — see "Known gaps" #2 |
| A bucket writer substitutes the *code* a job runs, not just its data (TB6) | **Accepted, and it is the sharpest edge in the sample** — the merge and eval jobs take their entrypoint from S3 and the training job restores a pickled checkpoint from S3, so `s3:PutObject` on three prefixes is root execution inside a GPU container holding the execution role. Nothing outside your account can reach it; a second in-account principal with bucket write can. See "Known gaps" #10 |
| The job role can delete the frozen base weights it only ever needs to read | **Accepted** — `bucket.grant_read_write(sm_role)` (`pipeline_stack.py:139`) is bucket-wide, so `s3:DeleteObject*` covers `checkpoints/` too. Re-staging is a ~128GB, multi-hour recovery. Split the grant if that matters to you: `grant_read` on `checkpoints/*`, read-write on the run-scoped prefixes |
| Poisoned or mislabelled training data | **Out of scope** — the validation stage catches structural defects (dim mismatches, `action == state` echo, frame/row misalignment), not adversarial content. Data governance is yours |
| Account identifiers leaking into this public repo (TB5) | **Mitigated** — all account-specific values live only in `pipeline/pipeline_config.json`, which is generated by the CDK stack and gitignored; `pipeline_config.example.json` is the committed template |

## Known gaps you must address before going to production

These are deliberate omissions, not oversights. Each one's correct remedy
depends on your environment, so the sample states the gap plainly instead of
guessing.

1. **The training container runs as `root`** (CKV_DOCKER_3, and the equivalent
   Semgrep non-root-user rule).

   - *Why it is unfixed here:* SageMaker's training platform mounts
     `/opt/ml/input`, `/opt/ml/model` and `/opt/ml/checkpoints` root-owned, and
     the checkpoint-sync agent that mirrors `/opt/ml/checkpoints` to S3 runs
     against those paths. A non-root `USER` needs every one of them chowned at
     build time and re-verified against the platform's mount behaviour.
   - *Compensating controls:* the image is never exposed as a network service —
     it has no listening port and no `CMD` other than the training entrypoint;
     it runs in a SageMaker-managed, single-tenant, ephemeral container that is
     destroyed when the job ends; and it holds no credential of its own beyond
     the execution role's temporary session.
   - *If your baseline requires a non-root container,* add a user and take
     ownership of the SageMaker paths before dropping privileges:

     ```dockerfile
     RUN groupadd -r dz && useradd -r -g dz -d /home/dz -m dz \
         && mkdir -p /opt/ml/input /opt/ml/model /opt/ml/checkpoints /opt/ml/code \
         && chown -R dz:dz /opt/ml /home/dz
     USER dz
     ENV HOME=/home/dz
     ```

     Then re-run the **smoke job** and confirm two things specifically: that
     `checkpoint-*` directories actually appear under
     `s3://<bucket>/sagemaker/checkpoints-sync/<job>/`, and that the merge
     stage succeeds. A permissions failure in the sync agent does not
     necessarily fail the job — you can get an exit-0 run with no usable
     checkpoint.

2. **S3 versioning and server access logging are off, and staged weights have
   no integrity manifest.** Versioning is off on purpose (the bucket holds
   hundreds of GB of write-once model shards; versioning every retrain balloons
   cost with no rollback value) and access logging with it. The consequence is
   concrete: a principal that holds `s3:PutObject` on the bucket could swap a
   base-weight shard, the next job would train against it without raising
   anything, and there would be no data-plane audit trail. For a regulated
   deployment, enable bucket versioning and server access logging (or
   CloudTrail data events), and consider S3 Object Lock or a recorded checksum
   manifest verified at job start. Both settings and their reasons are at
   `pipeline_stack.py:106-128`.

3. **Base image and OS-package currency.** The Dockerfile pins the SageMaker
   PyTorch DLC tag `2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker` and every
   dependency *this repo installs by name* — `deepspeed==0.16.9`,
   `transformers==4.51.3`, `flash-attn==2.8.3.post1` built from source — because
   that exact combination is the validated environment. Pinning is right for
   reproducibility and wrong for CVE exposure: **the image starts accumulating
   vulnerabilities the day you fork.** ECR scan-on-push is enabled so the
   findings will be there, but nothing in this sample acts on them. Rebuild and
   rescan on a schedule, and re-validate a smoke job after any pin bump — CVE
   remediation is the forker's responsibility.

4. **Most of the image's Python dependency tree is *not* pinned, and there is no
   lockfile or SBOM.** The pins in #3 are the ones this repo writes; they are the
   minority. `docker/Dockerfile:72` installs upstream DreamZero with
   `pip install -e .`, and that `pyproject.toml` (at the pinned commit
   `ab790c19`) declares **63 dependencies of which 38 carry no version specifier
   at all** — `hydra-core`, `pandas`, `matplotlib`, `timm`, `wandb`,
   `huggingface_hub`, `sentencepiece`, `tensorrt`, `nvidia-modelopt` and 29
   others. The Dockerfile re-pins exactly one of them (`deepspeed`, line 76), so
   **37 resolve to whatever PyPI serves at build time.** Two builds of the same
   commit can therefore contain different code, and a compromised or
   yanked-and-replaced release in any of those 37 enters the image with nothing
   to detect it. Reproduce the count yourself:

   ```bash
   ./setup.sh            # clones dreamzero/ at the validated pin
   grep -c '"' dreamzero/pyproject.toml    # then read the dependency arrays
   ```

   There is still no `requirements.lock`, no `--require-hashes` and no
   `constraints.txt`, so the build remains non-reproducible — consume the image
   **digest-pinned** (as `pipeline_config.json` and the CodeBuild factory both do)
   rather than by tag.

   **What has been closed: the inventory half.**
   [DEPENDENCY-INVENTORY.md](DEPENDENCY-INVENTORY.md) records a `pip-audit` scan of
   the built `v11` image (digest recorded there, scanned 2026-09-02) with the exact
   commands to reproduce it: **382 packages installed, 7 carrying 33 unique
   advisories.** It also triages reachability rather than reporting a count, and
   the headline result is one advisory that is **not** dismissible:

   > **PYSEC-2026-2288**, arbitrary code execution in `transformers`
   > `Trainer._load_rng_state`, fixed in transformers 5.0.0 — this image pins
   > 4.51.3. It is reachable because the restore path downloads
   > `rng_state_<N>.pth` as part of `checkpoint-<N>/` and the Trainer loads them.

   It grants **no privilege beyond TB6 / gap #10 above**: exploiting it requires
   `s3:PutObject` under `checkpoints-sync/<job>/`, which this document already
   states is equivalent to code execution because DeepSpeed unpickles restored
   shards with `weights_only=False`. Same precondition, same remedy — restrict
   write on those prefixes. It is not fixed by upgrading, because 4.51.3 is pinned
   for the DeepSpeed resume behaviour the restore path depends on; moving to 5.x
   changes that contract and needs the resume path re-validated.

   Still outstanding here: the scan covers **Python only**. Enable ECR enhanced
   scanning or run Trivy/Inspector for the OS layer, CUDA/cuDNN and the compiled
   flash-attn extension, and re-run the inventory against each digest you deploy —
   advisories appear against unchanged versions.

5. **CodeBuild runs in privileged mode** (`pipeline_stack.py:221`), which is
   required for docker-in-docker image builds. It is scoped as tightly as that
   allows: push only to this project's own ECR repo, pull only from AWS's
   public DLC account, write only its own SSM parameter, and it builds only the
   `docker/` directory shipped as a CDK asset. **There is no source webhook and
   no untrusted-PR path in this sample.** If you add one, re-evaluate: a
   privileged build running untrusted code is a container-escape path to the
   build role's credentials.

6. **Spend has no hard ceiling.** Per-stage `max_runtime_hours` caps a single
   job, and `--stop-after smoke` exists, but `run_pipeline.py` runs straight
   through train and merge by default and never prompts. Before handing this to
   a team, add an AWS Budgets alarm and, if you can, an SCP or IAM condition
   restricting `sagemaker:CreateTrainingJob` to the instance types you intend
   to pay for. The README's Costs section has measured per-stage numbers to
   size the budget against.

7. **Encryption uses service-managed keys, not CMKs.** S3 is SSE-S3, ECR is
   the AES-256 default, and jobs set no `VolumeKmsKeyId` or output KMS key. If
   you need key isolation, auditable key usage, or cross-account grants, switch
   the bucket and ECR repo to KMS keys and add the KMS parameters to the job
   submissions in `run_pipeline.py` and `evaluation/submit_eval_job.py` —
   at which point `AwsSolutions-CB4` becomes moot rather than suppressed.

8. **Jobs run outside your VPC, with no network isolation.** No `VpcConfig`
   and no `EnableNetworkIsolation` is set on any job, so containers run in the
   SageMaker-managed VPC with egress. The train and merge jobs do not need it:
   the image sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and all
   weights and data arrive through S3 input channels. The **eval** job is the
   exception, and it matters before you reach for the remedy:

   - `run_eval_in_job.sh:27` runs an unpinned `pip install awscli` as root.
     It is wrapped in `2>/dev/null || true`, so it is a no-op when the base
     image already ships the distribution and a silent failure when it does
     not — but it is still an attempt to resolve and execute a package from
     PyPI at job time, with no version or hash pin.
   - `run_eval_in_job.sh:108` uploads each arm's results with `aws s3 sync`.

   So `EnableNetworkIsolation=True` on the eval job does not merely stop the
   `pip` line, it breaks the **result upload** — the job's entire output. To
   isolate that job you need an S3 VPC endpoint its subnets can reach, and you
   should drop the `pip` line and let the image own the CLI. For train and
   merge, `VpcConfig` with private subnets, an S3 endpoint and
   `EnableNetworkIsolation=True` is a clean fit as shipped.
   (Inter-container traffic encryption is not applicable: every job in this
   sample is `InstanceCount: 1`.)

9. **ECR tags are mutable** (`pipeline_stack.py:93`), so that iterating on the
   image with `build_and_push.sh <tag>` works. Jobs are insulated from this by
   the digest pin, but if your baseline requires immutability, set
   `TagMutability.IMMUTABLE` and issue a new tag per build.

10. **Three prefixes turn bucket write into code execution** (TB6). The image is
    pinned by digest, but the *code the container runs* is not always in the
    image:

    - the merge job is submitted with
      `ContainerEntrypoint: ["python", "/opt/ml/input/data/lora/merge_lora.py"]`
      (`run_pipeline.py:279`), and the driver uploads that file to
      `sagemaker/lora-checkpoints/<job>/merge_lora.py` moments earlier
      (`run_pipeline.py:468-469` and `:497-498`);
    - the eval job is submitted with
      `ContainerEntrypoint: ["bash", "/opt/ml/input/data/evalscript/run_eval_in_job.sh"]`
      (`submit_eval_job.py:116`) and reads it from the `sagemaker/eval-assets/`
      prefix you upload by hand (`:60-67`);
    - the **training** job restores its own checkpoint mirror at start-up.
      `train_entrypoint.sh:532-546` syncs `sagemaker/checkpoints-sync/<job>/` (the
      prefix the driver sets at `run_pipeline.py:234`) into the output directory,
      and DeepSpeed 0.16.9 then unpickles those shards with
      `weights_only=False` — arbitrary code, by design of the pickle format.
      This is not new in kind: the manual `ckpt` input channel always resumed
      from the same prefix the same way. What changed in image v10 is that it is
      now **automatic** on every start, so it no longer takes an operator
      deciding to chain a resume. It is also the one entrypoint of the three that
      a *third party* never writes: only the training job itself and whoever
      staged a seed checkpoint put objects there.

    All three prefixes are unversioned (gap #2) and the object is referenced by
    key, not by `VersionId`. So any principal holding `s3:PutObject` on those
    prefixes chooses what runs as **root** on a GPU instance holding the
    execution role's credentials — which, per the `AwsSolutions-IAM4` row above,
    includes `AmazonSageMakerFullAccess`. That is a privilege boundary, not a
    data one, and it is not what a reader assumes from "jobs reference the image
    by digest."

    In the documented single-operator flow this is not reachable: the only writer
    is the same person who could submit the job directly. It becomes real the
    moment the bucket has a second writer — a CI role, a data-engineering
    pipeline, a teammate with `PowerUserAccess`. Before that happens, do one of:

    - **bake both scripts into the image** and drop `ContainerEntrypoint`
      entirely (removes the two script channels; costs you an image rebuild per
      script edit);
    - **pin the object version** — enable versioning on the bucket and pass the
      `VersionId` you just uploaded, so a later overwrite cannot change what a
      queued job runs (the same reasoning that already justifies the ECR digest
      pin); or
    - **restrict all three prefixes** with a bucket policy or SCP so only the
      deploying principal and the execution role can `PutObject` there, and audit
      them with CloudTrail data events.

    Note that the first two remedies do not reach the checkpoint prefix. Baking
    the scripts leaves it, and versioning does not help either, because the
    restore picks the newest complete `checkpoint-*` rather than a version you
    name. Resuming from a checkpoint *is* loading a pickle; the only real
    mitigations are keeping write access to that prefix down to the job's own
    role, or accepting a from-scratch restart instead of a resume. If you would
    rather have the restart, unset `checkpoint_s3_uri` on the training job:
    `train_entrypoint.sh:245` makes the restore path a no-op without it. Weigh
    that properly, though — the same variable drives the upload loop, so
    unsetting it means no checkpoint ever leaves the instance, and an interrupted
    spot job loses **all** of its progress rather than the last few hundred
    steps.

---

Found a security issue in this sample? Please **do not** open a public issue —
see [CONTRIBUTING](../CONTRIBUTING.md#security-issue-notifications) for the
reporting process.
