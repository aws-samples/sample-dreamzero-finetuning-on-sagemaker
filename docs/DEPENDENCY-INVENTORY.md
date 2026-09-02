# Dependency inventory and CVE scan

Closes the gap [SECURITY.md](SECURITY.md) previously listed as open: this repo
declares dependencies without upper bounds and shipped no software-composition
inventory, so nobody could say what is actually installed or what is known
vulnerable in it.

**This scan is of the built image, not of the requirements files.** That
distinction matters: the requirements pin a handful of direct dependencies, but
the image also inherits everything in the SageMaker Deep Learning Container base
and everything upstream DreamZero's own `pip install -e .` resolves. Scanning the
declared dependencies would have missed most of what runs.

## Method — reproducible as-is

```bash
IMG=<account>.dkr.ecr.<region>.amazonaws.com/dreamzero-sagemaker-training@sha256:<digest>
docker run --rm --entrypoint bash "$IMG" -c 'pip list --format=json'
docker run --rm --entrypoint bash "$IMG" -c \
  'pip install -q pip-audit && pip-audit --format=json --progress-spinner=off'
```

| | |
|---|---|
| Image scanned | `dreamzero-sagemaker-training@sha256:506f7918c05d8ffcd6e34a564cb1ed4d271b7f77e599f0aa76d212528829d755` (image tag `v11`) |
| Scanned | 2026-09-02 |
| Tool | `pip-audit` against PyPI + OSV advisory data |
| Packages installed | **382** |
| Distributions audited | 395 (audit counts some distributions `pip list` collapses) |
| Packages with advisories | **7** |
| Unique advisories | **33** |

`pip-audit` exits non-zero when it finds anything, so a `0` exit here would mean
a clean image, not a successful scan. It exited `1`.

## What was found

| Package | Version | Advisories | Examples | Fix available upstream |
|---|---|---|---|---|
| `transformers` | 4.51.3 | 18 | PYSEC-2025-217, PYSEC-2025-214, PYSEC-2025-218 … | yes — 4.52.1 |
| `ray` | 2.47.1 | 6 | PYSEC-2026-517, PYSEC-2026-520, PYSEC-2026-518 … | yes — 2.52.0 |
| `diffusers` | 0.30.2 | 3 | PYSEC-2026-2446, PYSEC-2026-40, PYSEC-2026-41 | yes — 0.38.0 |
| `opencv-python` | 4.8.0.74 | 2 | PYSEC-2023-183, GHSA-qr4w-53vh-m672 | yes — 4.8.1.78 |
| `tornado` | 6.5.7 | 2 | GHSA-wwv5-g3v4-889x, GHSA-8423-8fgw-73vq | yes — 6.5.8 |
| `datasets` | 3.6.0 | 1 | PYSEC-2026-3716 | yes — 5.0.1 |
| `pip` | 26.1.2 | 1 | PYSEC-2026-3721 | yes — 26.2 |

A count is not a risk assessment. Most of these sit in code this pipeline never
executes, and one does not.

## The one advisory on a path this repo exercises

**PYSEC-2026-2288 — arbitrary code execution in `transformers` `Trainer`
`_load_rng_state`** (fixed in transformers 5.0.0; this image pins 4.51.3).

It is reachable here because the resume path deliberately handles those files.
`docker/train_entrypoint.sh` size-checks `rng_state_<N>.pth` as part of judging a
checkpoint complete (`SIZE_CHECKED`, line 319) and downloads them with the rest of
`checkpoint-<N>/`; on resume the HuggingFace `Trainer` then `torch.load`s them.

**It grants no privilege this repo does not already document as accepted.**
Reaching it requires writing objects under
`s3://<bucket>/sagemaker/checkpoints-sync/<job>/`, and SECURITY.md already states
that write access to that prefix is equivalent to code execution in the training
job, because DeepSpeed 0.16.9 unpickles restored shards with
`weights_only=False`. Anyone who can plant a malicious `rng_state_*.pth` can
already plant a malicious optimizer shard. The remedy is the same and is already
prescribed: restrict `s3:PutObject` on those prefixes to the training role.

**Not fixed by upgrading, for a stated reason.** transformers 4.51.3 is pinned
because the DeepSpeed resume path depends on its behaviour — `scheduler.pt` is
loaded with no `os.path.isfile` guard in the DeepSpeed branch, which is why the
restore treats that file as mandatory. Moving to 5.x is a behavioural change to
the resume contract, not a version bump, and would need the whole restore path
re-validated. Deliberate decision: disclose and scope, do not silently upgrade.

## Why the rest are assessed as not reachable

Stated so a reviewer can disagree with specific reasoning rather than a verdict.

- **`ray` (6)** — every advisory targets the Ray dashboard, its job-submission
  API, or `ray.data.read_webdataset`. This pipeline starts no Ray cluster, exposes
  no dashboard, and calls no Ray API; `ray` arrives as a transitive dependency.
- **`transformers`, the other 17** — eight are `convert_*` scripts for unrelated
  model families (X-CLIP, SEW, SEW-D, GLM4, Perceiver, Transformer-XL,
  megatron_gpt2, HuBERT) that nothing here imports. Six are ReDoS in tokenizers
  and optimizers outside this recipe's path. The remainder require loading a model
  from an attacker-controlled repository; all weights here come from the
  operator's own S3 prefixes, staged by `pipeline/stage_base_assets.py` from
  revision-pinned sources.
- **`diffusers` (3)** — `trust_remote_code` bypasses in
  `DiffusionPipeline.from_pretrained`. This recipe loads Wan2.1 components from
  local `.pth` files at fixed paths, not via `from_pretrained` against a remote
  repo id.
- **`opencv-python`, `tornado`, `datasets`** — a bundled `libwebp` decode bug, two
  HTTP-server parsing bugs in a server that is never started, and a path traversal
  in folder-based dataset builders that this pipeline does not use (it consumes
  pre-converted LeRobot/GEAR trees).
- **`pip` (1)** — build-time only, and the image build resolves from PyPI over
  HTTPS with no custom index.

## Standing limitations

1. **Point-in-time.** New advisories appear against unchanged versions; this table
   was true on 2026-09-02 and will drift. Re-run the two commands above against
   whatever digest you deploy.
2. **Python only.** `pip-audit` does not cover the OS packages in the DLC base
   layer, CUDA/cuDNN, or the compiled flash-attn extension. An OS-level scan
   (ECR enhanced scanning, Inspector, Trivy) is complementary and not done here.
3. **Unbounded dependency declarations remain.** This inventory records what one
   resolution produced; it does not make the build reproducible. A future rebuild
   can install different versions, which is exactly why the image URI should be
   consumed digest-pinned — as `pipeline/pipeline_config.json` and the CodeBuild
   factory both do.
