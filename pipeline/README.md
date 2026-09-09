# Reusable pipeline: LeRobot dataset → servable DreamZero weights in S3

One command chains the validated stages:

```
dataset (HuggingFace repo id or local LeRobot v2.1/v3) + embodiment config
  → [fetch]    download from HuggingFace (skipped for local paths)
  → [detect]   read codebase_version
  → [convert]  v3 → v2.1 if needed (--v3-converter; layouts are dataset-specific)
  → [validate] dataset vs config: slice bounds, cameras, fps, completeness,
               action-vs-state
  → [prep]     v2.1 → GEAR/yam  (prep_dataset.py, config-driven)
  → [stage]    → s3://…/sagemaker/datasets/<name>/
  → [smoke]    SageMaker job, max_steps=10 — ~$10 gate before the ~$93 run
  → [train]    SageMaker job, --max-steps (default 1000)
  → [merge]    LoRA→AgiBot merge job → s3://…/sagemaker/models/<name>-merged/
```

## Usage

```bash
# defaults (dataset, compute, hyperparameters) come from project_config.json:
python3 pipeline/run_pipeline.py --name aloha-demo

# or point at your own data explicitly (the repo's dataset/ folder is the
# gitignored home for local data; HF downloads land there too):
python3 pipeline/run_pipeline.py \
    --dataset dataset/my_robot \
    --name myrobot-v1 \
    --config pipeline/configs/aloha_bimanual_14dim.yaml
# options: --hf-dataset REPO_ID  --hf-revision REV  --max-steps N
#          --skip-smoke  --skip-validation  --start-at <stage>  --dry-run
#          --stop-after <stage>  --project-config <json>
#          --v3-converter 'python3 conv.py {src} {dst}'  --skip-preflight
```

`--stop-after smoke` exits after the smoke job so you can inspect its loss and
per-step timing (and extrapolate the full run's cost) before committing;
resume with `--start-at train`.

**Size the smoke job to measure, not just to pass.** The 10-step default proves
the job runs, but the first few steps are dominated by warmup and shard
caching, so they overstate the step time. Set `compute.smoke.steps` to ~30 and
add `logging_steps=1` to `extra_overrides` — that leaves ~20 steady-state
per-step timestamps, enough for a median seconds/step you can multiply by
`max_steps`. On a new dataset this is the difference between discovering the
real cost for the price of one smoke job and discovering it after a job has been
billing for a day. A 30-step smoke costs about **$10 on the shipped
`ml.g7e.24xlarge`** and about **$25 on the `ml.g7e.48xlarge`** the
`project_config.openarms.json` recipe uses — both dominated by the ~20 minutes of
fixed start-up, not by the steps.
By default the smoke saves once, at its last step; set `compute.smoke.save_steps`
to a divisor of `steps` (e.g. steps 30, save_steps 10) to also exercise
*mid-run* checkpoint saves — the full save → S3-sync path a long run will
depend on — before committing to it.

## Hold out episodes for evaluation

Open-loop eval on episodes the model trained on says nothing about
generalization. `split_dataset.py` splits a v2.1 dataset into train/holdout
trees (renumbered contiguously — the GEAR converter requires it):

```bash
python3 pipeline/split_dataset.py --src /path/to/v21 \
    --holdout 3,17,42 --train-dst /path/to/v21_train --holdout-dst /path/to/v21_holdout
```

Run the pipeline on the train tree; prep + stage the holdout tree separately
(`prep_dataset.py`, then `aws s3 sync` to `datasets/<name>-holdout/`) and pass
that prefix to `evaluation/submit_eval_job.py --dataset-s3`.

Output contract (S3, per the S3-first asset policy):
- `models/<name>-merged/` — **servable** merged checkpoint (use this)
- `lora-checkpoints/<train-job>/` — raw LoRA, a few hundred MB (~220MB at the
  shipped defaults; scales with LoRA rank) — archival/re-merge
- `checkpoints-sync/<train-job>/loss_log.jsonl` — live loss during training

## The embodiment config (the part you must think about)

Per-robot semantics the pipeline cannot infer — wrong values train garbage
*without erroring*:

| Field | Meaning | Mistake mode |
|---|---|---|
| `camera_map` | source camera → yam view (top=scene, left/right=wrist) | swapped views: model sees the wrong 2×2 grid |
| `state_keys`/`action_keys` | packed-vector slices | mis-sliced dims: silent garbage actions |
| `task_key` | language annotation column | crash (best case) |
| `fps` | dataset frame rate; cross-checked against the dataset's own meta, forwarded as `fps.yam` | a mismatch means the config was copied from a different robot — treat it as a proxy for the errors above |

On `fps` specifically: the value is a consistency check, not a training knob.
The trainer's data recipes all pin the **decord** video backend, which selects
frames by matching the parquet `timestamp` column to the video's own
presentation timestamps and does frame-index arithmetic for action chunks —
the configured fps is never read on that path (it would matter only under
`video_backend=torchcodec`). The invariant that actually breaks training
silently is **video-frame ↔ parquet-row alignment**: frame *i* of each video
must correspond to row *i* of the episode's parquet, which the prep/GEAR
conversion establishes and re-encoding must preserve (prep's h264 re-encode
keeps every frame, so it does).

`configs/aloha_bimanual_14dim.yaml` is the validated default for ALOHA-like
14-dim bimanual data.

## What is validated

- **The prep is exact.** `prep_dataset.py` with the ALOHA config reproduces
  the GEAR dataset the reference run trained on byte-for-byte — all meta JSONs
  parsed-equal, both jsonl exact, all parquets md5-identical, same video tree.
  That dataset produced the 0.0957-loss run and the 0.00106-MSE checkpoint.
- **The full chain has run end to end** (fetch → validate → prep → stage →
  smoke → train → merge) with paid jobs; the root README's headline numbers
  come from that run. On a new dataset, still watch the smoke gate before
  letting the full run proceed.

## Caveats

- **Keep `per_device_train_batch_size` at 1; raise the effective batch with
  `global_batch_size` instead.** Upstream only ever ran batch size 1, and
  three independent problems hide behind that. (1) The video sampler can emit
  a 3-chunk sample where a chunk window touches the episode end; the collator
  `np.stack`s samples, so a batch mixing 3- and 4-chunk samples crashes with
  `all input arrays must have the same shape` — fixed by
  `docker/patches/0002` (every sample exactly `max_chunk_size` chunks).
  (2) The action-head loss multiplies by `has_real_action[:, None]`, a
  `(B, 1)` tensor whose right-aligned broadcast only lines up at `B == 1`;
  at batch 2+ every rank fails deterministically on step 0 — fixed by
  `docker/patches/0003`. (3) Even with both patches, **batch 2 OOMs on 96GB
  GPUs**: the 14B DiT forward at 33 frames × 3 views peaks at ~94 of 95 GiB
  per GPU at batch 1-equivalent load and dies asking for ~1.6 GiB more.
  The supported way to scale the effective batch is gradient accumulation:
  add `global_batch_size=<N>` to `extra_overrides` — the trainer derives
  `gradient_accumulation_steps = N / (batch × world_size)` automatically.

- **Host RAM, not just GPU RAM, can kill a job during shard caching.** The
  sharded loader decodes an entire shard of video into host memory *per
  dataloader worker* — roughly `rows × cameras × H×W×3` bytes, and about double
  that while the next shard prefetches. The fingerprint: ~32 `Caching shard`
  lines, then `DataLoader worker (pid …) is killed by signal: Killed` and the
  job fails with SageMaker's `ClientError: Please use an instance type with
  more memory…`. The GPUs are innocent. Two knobs bound it, both already set
  in the shipped `project_config.json`: `dataloader_num_workers` (per GPU) and
  `++train_dataset.dataset_kwargs.num_steps_per_shard=<rows>` in
  `extra_overrides` (smaller shards = smaller caches, more of them). Budget
  `ranks × workers × shard_rows × cameras × frame_bytes × 2` against the
  instance's RAM before raising either.

- v3→v2.1 conversion is **not universal** — v3 aggregates episodes into shared
  files in dataset-specific ways. On HuggingFace, try the `v2.1` tag before
  writing one: `lerobot/*` repos generally still tag a v2.1 tree even though
  `main` now points at v3, which is why the shipped `project_config.json` pins
  `dataset.revision: "v2.1"`. Pass `--hf-revision v2.1` alongside
  `--hf-dataset` — it is only read on that path, so for a `repo_id` set in the
  config, edit `dataset.revision` instead. Only when no v2.1 tree exists,
  convert locally with `convert_lerobot_v3_to_v21.py` — a worked example for
  bimanual 3-camera v3 trees that splits the aggregated parquet on
  `episode_index`, cuts frame-exact clips, and regenerates the v2.1 meta. Run it
  standalone and point `dataset.local_path` at its output, or inline via
  `--v3-converter 'python3 pipeline/convert_lerobot_v3_to_v21.py {src} {dst}'`.
  A materially different layout needs its split/cut logic adapted.
- Non-ALOHA embodiments need a new config; if state/action isn't a packed
  14-dim yam-order vector, you also need column re-packing (not built — the
  converter only slices, it doesn't reorder).
- The smoke gate asserts job completion, not loss quality; eyeball
  `checkpoints-sync/<job>/loss_log.jsonl` before letting the full run proceed
  if the dataset is new. It applies the same finalization-failure tolerance the
  train stage does (see below), so a gate failure means the *smoke* failed.
- The validated runs used `ml.g7e.24xlarge`. On instance types where SageMaker
  attaches an **EFA** adapter (`ml.p4de.24xlarge`, p5-class), the dataloader's
  `fork()` aborts every rank with `SIGABRT` unless `FI_EFA_FORK_SAFE=1` is set
  — the entrypoint exports it, harmlessly, on every instance type. Note that a
  type reporting `EfaSupported=true` is not the same as SageMaker attaching an
  adapter: g7e reports true but gets none, which is why the validated g7e runs
  never hit this. It fires ~5 min in, after weight loading and a clean NCCL
  init, so it reads like a training bug; filter `NCCL INFO`/`NET/OFI` out of
  the CloudWatch log to see the real message.
- **Checkpoints go through SageMaker's `CheckpointConfig`.** The launcher sets
  `LocalPath=/opt/ml/checkpoints` and `S3Uri=checkpoints-sync/<job>/` on every
  training job; SageMaker mounts the path on the data volume, mirrors each save
  to S3 as the trainer writes it, and on a managed-spot relaunch restores the
  prefix before the container starts. Upstream's trainer resumes from the
  highest `checkpoint-*/` on its own. What the entrypoint adds is a
  completeness guard in front of it: one optimizer shard per GPU on this
  instance plus `latest`, `trainer_state.json`, `scheduler.pt`, the adapter and
  its config, no zero-byte file, and sizes consistent across the run's
  checkpoints. A checkpoint torn by a mid-save reclaim is deleted so the trainer
  falls back to the previous one rather than failing in `torch.load`. The guard
  then reads the newest checkpoint's `global_step`: at or past `max_steps` (a
  reclaim between the last save and teardown) the finished weights stay and the
  trainer skips to staging them; below it (a deliberate continuation with a
  higher `max_steps`) the finished run's top-level `config.json` is removed,
  because upstream reads it as "skip training" and would report SUCCESS having
  trained nothing. Without
  that guard a spot restart can be a data-loss event, not just a cost one: a
  relaunch that trains from step 0 re-uploads `checkpoint-N/` over the **same
  keys** in a non-versioned bucket, destroying the better-trained recovery
  point — so the guard never fails open. Checkpoints are ~3.6 GB each
  (`docker/patches/0004`), so the mirror is cheap and `/opt/ml/checkpoints`
  never fills.

- **A job that failed *after* training succeeded is still recovered
  automatically.** It is rare, but SageMaker's post-container finalization can
  fail independently of anything the container did, so the recovery machinery
  stays in as defense-in-depth. Recognizing it: the failure lands seconds after the
  container *already exited cleanly* — if `=== done ===` is in the log,
  torchrun returned 0 and `model.safetensors` was verified present in
  `/opt/ml/model`, so nothing in your code or data is at fault. There is no
  Python traceback, and `FailureReason` is the generic string rather than
  `ENTRYPOINT FAILURE: …` (a real container-side fault surfaces through
  `/opt/ml/output/failure`, so it would say so). If the job's sync was still
  draining when it failed, expect a **truncated** `checkpoints-sync/<job>/`
  alongside it: the trainer writes a
  checkpoint's `model.safetensors` and `config.json` *first* and its DeepSpeed
  resume states last, so the truncation eats resume states while the LoRA
  weights — the only thing the merge needs — land intact minutes earlier.

  The recovery: `model.tar.gz` is written only for a job that reaches
  `Completed`, so `wait_for_job` doesn't treat this `Failed` as fatal —
  `finalization_only_failure` re-lists `checkpoints-sync/<job>/` and, if
  `checkpoint-<max_steps>` — the *final* save — is complete, logs why and lets
  the run continue into merge; `stage_lora_for_merge` then rebuilds
  `lora-checkpoints/<job>/` from that checkpoint (server-side copies, so the
  weights never round-trip). The **smoke gate uses the same tolerance**, keyed
  on the smoke's own step count. The gate is deliberately the final checkpoint
  rather than "any complete checkpoint": a job that died at step 3000 of 7000
  also leaves complete earlier checkpoints, and quietly merging a half-trained
  model is worse than stopping — a genuine mid-run failure still aborts, and
  prints which checkpoints *are* mergeable so you can decide. `Stopped` never
  auto-continues — a runtime cap or an external `StopTrainingJob` says nothing
  about training having finished.

  If you do end up driving it by hand, run `--start-at merge --train-job <job>`
  after confirming the run was healthy in `checkpoints-sync/<job>/loss_log.jsonl`.
  **Don't resubmit a long run over a post-training failure** — the training
  itself finished and its checkpoints are already in S3, so a retry only buys
  the same hours again; merge what you have.

- Lost the terminal during a multi-day run? The SageMaker job is server-side and
  does not care, but the local process is what carries the run into merge. Re-
  attach with `--start-at train --train-job <job>`: the train stage waits on the
  job you name instead of submitting a second copy of it, and the pipeline then
  proceeds exactly as if it had never been interrupted. Editing `run_pipeline.py`
  does *not* affect an already-running process, so this is also how you pick up a
  fix mid-run — kill the old poller first, or two of them will each launch a
  merge job and race for the same GPU.

- 80GB+ GPUs are scarce; jobs routinely sit in `InProgress/Pending —
  "Training job waiting for capacity"` for hours. That is normal and costs
  nothing, so don't cancel and resubmit — you lose your place in the queue.
  But if your account runs cost-cleanup tooling that stops resources on a
  schedule, it will happily stop a job that is still queued: the job ends
  `Stopped` with **no `FailureReason`**, indistinguishable from a manual
  cancel. Set that tooling's exemption tag in `extra_job_tags`
  (`pipeline_config.json`) before submitting anything long-queued, and check
  CloudTrail for `StopTrainingJob` if a job dies without explanation.

- Adding an AWS call? Build the client with `solution.get_client("s3",
  session=boto_session(cfg))`, not `boto_session(cfg).client("s3")`. The
  solution user-agent suffix is attached when the client is constructed, so a
  bare `.client()` call still works and is simply never attributed — nothing
  errors, which is exactly why it is easy to get wrong. `get_client` merges into
  any `Config` you pass, so timeouts and retry settings survive. Code that runs
  inside a job container has no repo to import from and reads the
  `USER_AGENT_STRING` environment variable instead (see `merge_lora.py`); the
  AWS CLI has no equivalent hook at all, so `aws s3 sync` in the entrypoints
  cannot be attributed. `pipeline/tests/test_solution.py` asserts on the real
  outbound header rather than on the config object, and is worth re-running
  after touching any of this.
