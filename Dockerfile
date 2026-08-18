# Same base and layering strategy as the LTX-2.3 worker, which is proven on this
# endpoint. The CUDA wheels installed below carry their own runtime, so the
# base image's cu121 toolkit only supplies the build tools, not the runtime.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Download via the classic LFS/HTTP path rather than HuggingFace's Xet backend.
# hf_xet dies parsing content hashes ("Unable to parse string as hex hash value")
# against Xet-backed repos; the LFS path fetches the same files without incident.
ENV HF_HUB_DISABLE_XET=1
ENV MODELS_ROOT=/workspace/models

# Runtime defaults, tuned for the L40S (Ada, SM 8.9, 48 GB). Declared here rather
# than only in handler.py so `docker inspect` shows what the image will do, and
# so a RunPod endpoint env var visibly overrides a stated default.
#
#   fp8-cast   stores the covered transformer Linears as float8_e4m3fn and
#              upcasts to bf16 inside forward() -- no _scaled_mm, no capability
#              floor. ~44 GB of bf16 weights becomes roughly 26-30 GB.
#              Do NOT switch this to fp8-scaled-mm: it needs a pre-quantized fp8
#              checkpoint and LTX-2.5 publishes bf16 transformers only.
#   cpu        bounds peak VRAM at the streaming buffer, so this pair cannot OOM.
#              Try LTX_OFFLOAD_MODE=none next; it removes streaming entirely and
#              should fit in 48 GB at 768x512x49.
#   conv       convolutional video VAE. The diffusion decoder needs natten, which
#              pins torch==2.13.0+cu132 against the pins below.
#
# Rollback if fp8-cast misbehaves: set LTX_QUANTIZATION=none on the endpoint.
ENV LTX_QUANTIZATION=fp8-cast
ENV LTX_OFFLOAD_MODE=cpu
ENV LTX_VIDEO_VAE=conv

# Install system dependencies (ffmpeg is required for video/audio encoding).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    pkg-config \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Pin the torch stack before installing ltx-*. The base image ships torch and
# torchaudio 2.4.0; ltx-core requires torch~=2.7, so pip would upgrade torch but
# leave torchaudio untouched (its bare requirement is already satisfied). The
# stale torchaudio is then built against a libtorch it no longer has, and
# importing it dies on an undefined torch::autograd symbol.
#
# These are the versions the 2.3 worker runs. They satisfy every constraint
# ltx-core 1.2.0 declares (torch~=2.7), so this is deliberately the smallest
# possible change from a known-good image -- see the note on LTX_REF below.
#
# Installed from the CUDA 12.8 index, not plain PyPI. This is a hardware
# requirement, not a preference: Blackwell needs a CUDA 12.8+ build to carry
# sm_100/sm_120 SASS. PyPI's default torch wheel has historically been an older
# CUDA variant, and one without sm_120 still imports, still reports the GPU, then
# dies on the first real kernel with "no kernel image is available for execution
# on the device" -- or JITs from PTX slowly enough to look like a hang.
#
# Harmless on Ada and Ampere (an L40S or A40 finds sm_89/sm_86 in the same wheel),
# so this one index covers every GPU this worker has been pointed at.
# handler.check_torch_supports_arch() verifies the result at boot.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} \
    torch==2.9.1 \
    torchaudio==2.9.1 \
    torchvision==0.24.1

# transformers is the single most fragile dependency in this stack and the one
# that changed most between 2.3 and 2.5. ltx-core 1.2.0 declares
# transformers>=5.8.0,<5.15 -- the floor is Gemma 4 support and the ceiling is
# real: 5.15.0 routes config attribute access through a heterogeneity layer that
# raises on global `config.head_dim`, while its own gemma4_unified model code
# still reads it that way, so the LTX-2.5 text encoder cannot be built at all.
# Pinned explicitly rather than left to the resolver because upstream deleted
# its uv.lock at the 1.2.0 release, so nothing else holds this in place.
RUN pip install --no-cache-dir "transformers>=5.8.0,<5.15"

# Install the LTX packages from the monorepo, pinned to a commit. 400fd31 is
# main at the v1.2.0 release (the first release with LTX-2.5 support: Gemma 4,
# split checkpoints, ModelPaths, and the ancestral stage-1 sampler).
#
# Pin this, never track a branch: upstream ships breaking API changes inside
# minor releases -- TilingConfig.default() and the DistilledPipeline constructor
# both changed in 1.2.0 -- and a floating ref turns those into a broken image
# on an unrelated rebuild.
ARG LTX_REF=400fd31
RUN pip install --no-cache-dir "git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-core"
RUN pip install --no-cache-dir "git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-pipelines"

# Serverless dependencies. Deliberately not installed:
#   natten      -- only speeds up the *diffusion* video VAE, and pins
#                  torch==2.13.0+cu132, which would fight the pins above. This
#                  worker defaults to the conv VAE (LTX_VIDEO_VAE=conv), which
#                  needs none of it.
#   ltx-kernels -- required only for nvfp4 quantization on Blackwell. Compiles
#                  CUDA extensions at install time and needs a matching nvcc.
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    huggingface_hub \
    av \
    tqdm \
    pillow \
    openimageio \
    "cloudpickle>=3.1"

WORKDIR /app

COPY rp_handler.py handler.py download_models.py staging.py model_paths_config.py ./

# Expose the target models path, plus the local-disk cache the handler stages
# weights into. /local-cache must live on the container disk, never on the
# network volume -- staging exists precisely to get the reads off that volume.
RUN mkdir -p /workspace/models /local-cache

# Fail the build if the API surface this handler depends on has moved. Cheap
# insurance: every one of these imports changed location or signature between
# LTX-2.3 and 2.5, so a bad LTX_REF is caught here rather than on a cold start
# after RunPod has already pulled a 20 GB image.
RUN python -c "\
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number; \
from ltx_core.model.video_vae.transformer import DiffVAEMode; \
from ltx_pipelines.distilled import DistilledPipeline; \
from ltx_pipelines.utils.model_paths import ModelPaths; \
from ltx_pipelines.utils.quantization_factory import QuantizationKind; \
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, OffloadMode; \
from ltx_pipelines.utils.args import ImageConditioningInput; \
from ltx_pipelines.utils.media_io import encode_video; \
import inspect; \
sig = inspect.signature(DistilledPipeline.__init__); \
assert 'model_paths' in sig.parameters, 'DistilledPipeline no longer takes model_paths -- LTX_REF is wrong'; \
print('LTX-2.5 API check passed')"

# Entrypoint is rp_handler.py, matching RunPod's reference worker layout. It
# calls handler.boot() then runpod.serverless.start(); handler.py holds the
# implementation and is import-safe on its own.
CMD ["python", "-u", "rp_handler.py"]
