# DreamZero LoRA fine-tuning — SageMaker BYOC training image.
#
# Reproduces the validated reference environment:
#   torch 2.8.0+cu129 / python 3.12  (from the SageMaker PyTorch DLC)
#   dreamzero @ ab790c1 (upstream main used by the EC2 run)
#   deepspeed 0.16.9   (0.19.x circular-imports vs pinned transformers 4.51.3)
#   flash-attn 2.8.3.post1 built from source at image-build time (no GPU needed)
#     Verified cubins in the built wheel: sm80, sm90, sm100, sm120 — i.e.
#     A100 80GB (ml.p4de), H100 (ml.p5), H200 (ml.p5e/p5en), and Blackwell
#     (ml.g7e) all run. flash-attn's setup.py emits its own -gencode flags on
#     top of TORCH_CUDA_ARCH_LIST below, which is why sm80/sm100 are present.
# The DLC registry account (763104351884) publishes the same image per-region;
# build_and_push.sh passes DLC_REGION from pipeline_config.json so the docker
# login and this FROM always target the same registry.
#
# checkov:skip=CKV_DOCKER_3: SageMaker training containers run as root — the
#   platform mounts /opt/ml (input channels, model, checkpoints) root-owned,
#   and the checkpoint sync agent requires it. This image is never exposed as
#   a network service.
# checkov:skip=CKV_DOCKER_2: HEALTHCHECK applies to long-running services;
#   SageMaker training jobs are batch processes and never consult it.
ARG DLC_REGION=us-east-1
FROM 763104351884.dkr.ecr.${DLC_REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

ARG DREAMZERO_REPO=https://github.com/dreamzero0/dreamzero.git
ARG DREAMZERO_COMMIT=ab790c198fbce33503358efbbd4187ce9a89adf3

RUN apt-get update && apt-get install -y --no-install-recommends \
        linux-libc-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code
RUN git clone ${DREAMZERO_REPO} dreamzero \
    && cd dreamzero \
    && git checkout ${DREAMZERO_COMMIT}

# CRITICAL FIX #5 (patches/0001): the sharded loader seeds its within-shard
# shuffle RNG with the bare training seed, identically on every rank and
# dataloader worker. On a small dataset (few shards) two workers routinely
# process the same shard at the same schedule position and then yield
# IDENTICAL batches — silently shrinking the effective batch diversity. The
# patch mixes rank/worker into that seed and logs the first few (traj, step)
# samples per worker so cross-rank duplication is auditable from the job log.
#
# CRITICAL FIX #6 (patches/0002): the video sampler accepts a chunk anchored
# 24 frames before the episode end (anchor+23 fits) but its "+3 extra frame"
# then falls out of bounds, so the sample is trimmed from 4 to 3 chunks while
# num_frames/frame_seqlen are configured for exactly 4. Any batch mixing 3-
# and 4-chunk samples crashes np.stack in the collator ("all input arrays
# must have the same shape") and shape-mismatches inside the action head —
# with per_device_train_batch_size>1 this kills the job within the first
# steps. The patch aligns the boundary condition with the action/state
# samplers and skips (rather than yields) samples that cannot reach the full
# chunk count.
#
# CRITICAL FIX #7 (patches/0003): the action-head loss multiplies the (B, T, D)
# flow-matching loss by has_real_action[:, None] — a (B, 1) tensor that
# broadcasts from the right, so it only lines up when B == 1. Every upstream
# run used batch size 1, hiding it; at per_device_train_batch_size >= 2 every
# rank fails deterministically on the first step with "The size of tensor
# a (B) must match the size of tensor b (T)". Fixed to [:, None, None].
COPY patches/ /opt/ml/code/patches/
RUN cd dreamzero && git apply /opt/ml/code/patches/*.patch

WORKDIR /opt/ml/code/dreamzero
# The extra index is PyTorch's official wheel host (CUDA 12.9 builds), the
# documented install path; all other packages resolve from PyPI as usual.
# Keep the nosemgrep marker on the line directly above the RUN: semgrep only
# honours it on the same or the immediately preceding line, so any comment
# between the two silently disables the suppression.
# nosemgrep: dockerfile.audit.dockerfile-pip-extra-index-url.dockerfile-pip-extra-index-url
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu129

# CRITICAL FIX #1: pip resolves deepspeed 0.19.x, which
# circular-imports against transformers==4.51.3. Must come AFTER `pip install -e .`.
RUN pip install --no-cache-dir deepspeed==0.16.9

# CRITICAL FIX #2: flash-attn from source. TORCH_CUDA_ARCH_LIST lets
# it compile without a GPU in the build environment.
ENV TORCH_CUDA_ARCH_LIST="9.0;12.0"
RUN pip install --no-cache-dir ninja packaging \
    && MAX_JOBS=16 pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3.post1

# The DLC ships a system cuDNN in /usr/local/cuda/lib64 whose sublibrary
# builds differ from the pip nvidia-cudnn-cu12 copy torch links against.
# With both on the loader path, cuDNN's dispatcher and its dlopen'd
# sublibraries resolve from different copies and conv kernels die with
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED (seen on the first smoke job).
# Keep only the pip copy — the same single-copy layout as the EC2 venv.
RUN rm -f /usr/local/cuda/lib64/libcudnn* && ldconfig

COPY train_entrypoint.sh /opt/ml/code/train_entrypoint.sh
RUN chmod +x /opt/ml/code/train_entrypoint.sh

ENV HYDRA_FULL_ERROR=1 \
    WANDB_MODE=disabled \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# SageMaker invokes the container as `docker run <image> train`; the
# entrypoint ignores that argument.
ENTRYPOINT ["/opt/ml/code/train_entrypoint.sh"]
