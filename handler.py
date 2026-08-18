"""RunPod serverless handler for LTX-2.5 distilled text/image-to-video.

Serves DistilledPipeline on the LTX-2.5 split pack: the distilled transformer
plus the LTX-tuned Gemma 4 text encoder, video VAE, audio VAE and x2 spatial
upsampler. Video and synchronized audio are rendered to mp4 and uploaded to a
DigitalOcean Spaces bucket supplied per request.

Request contract is unchanged from the LTX-2.3 worker, so existing callers do
not need to move: prompt and dimensions at the top level of `input`, bucket
credentials under `input.s3_config`.

Migration notes for anyone diffing this against the 2.3 handler -- the LTX-2.5
API changed in four ways that matter here:

  1. DistilledPipeline takes a `model_paths: ModelPaths` object instead of
     `distilled_checkpoint_path` + `gemma_root`. 2.5 is a split pack, so
     `ModelPaths.from_split` is the only correct constructor; `from_monolith`
     would point the VAE slots at the transformer file, which carries neither
     VAE config nor VAE weights.
  2. `__call__` returns four values, not two: (video, audio, num_frames,
     tiling_config). The extra two exist because 2.5 can resolve num_frames
     itself from the caption, and tiling is now auto-derived from the VAE's
     actual compression factors rather than assumed.
  3. `TilingConfig.default()` is gone. Pass AUTO_TILING and read the resolved
     config back out of the return tuple to feed get_video_chunks_number.
  4. Stage 1 uses an ancestral SDE sampler on 2.5+ checkpoints. This is detected
     from checkpoint metadata inside the pipeline (`should_use_ancestral_sampler`),
     not configured here -- there is nothing to pass.
"""

import gc
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
import uuid
import warnings

import runpod
import torch
from runpod.serverless.utils import rp_upload

from download_models import ensure_models
from model_paths_config import (
    AUDIO_VAE_PATH,
    DURATION_HEAD_PATH,
    MODELS_ROOT,
    SPATIAL_UPSAMPLER_PATH,
    TEXT_ENCODER_PATH,
    TRANSFORMER_PATH,
    VIDEO_VAE_PATH,
    VIDEO_VAE_VARIANT,
    staging_candidates,
)
from staging import stage_files

# LTX-2.5 imports. AUTO_TILING and DiffVAEMode are new in 2.5; the rest keep
# their 2.3 import paths.
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, OffloadMode

# Global pipeline cache. Cheap to hold: upstream defers the expensive weight
# loads to call time, so this object graph is mostly builders.
current_pipeline = None
current_quantization = None
current_offload_mode = None

# Where staged copies of the weights live. Must be on the container's local
# disk, not the network volume -- the whole point is to stop reading the
# checkpoints over the network. Staging is skipped automatically when the disk
# is too small, so this is safe to leave on.
LOCAL_CACHE_DIR = os.getenv("LOCAL_CACHE_DIR", "/local-cache")
STAGING_ENABLED = os.getenv("DISABLE_LOCAL_STAGING", "0") != "1"
STAGING_RESERVE_BYTES = int(float(os.getenv("STAGING_RESERVE_GB", "5")) * 1024 ** 3)

# Resolved at startup: {original_path: path_to_actually_load_from}. Falls back
# to identity so the handler works unchanged when staging is off or skipped.
STAGED_PATHS = {}

# quantization/offload_mode are deployment properties, not request properties:
# changing either rebuilds the pipeline, so honouring them per request lets one
# caller evict the warm pipeline for everyone queued behind it. Set
# ALLOW_REQUEST_PIPELINE_OVERRIDE=1 to accept them from the payload for an A/B.
#
# Picking them is a function of the GPU, and the two are coupled.
#
# THIS DEPLOYMENT TARGETS THE L40S (Ada Lovelace, SM 8.9, 48 GB).
#
# The two FP8 backends are not the same thing, and the LTX-2.3 worker's comments
# conflated them. Read ltx_core/quantization/ before trusting either:
#
#   fp8-scaled-mm  calls torch._scaled_mm (fp8_scaled_mm.py:59), which needs
#                  native FP8 tensor cores -- SM 8.9+. The L40S IS SM 8.9, so
#                  the hardware qualifies. It is still NOT usable here, for a
#                  different reason: get_fp8_swap_module_ops raises unless the
#                  checkpoint ships F8_E4M3 .weight plus a sibling .weight_scale,
#                  and LTX-2.5 publishes bf16 transformers only (the README lists
#                  exactly two, both -bf16). Upstream's own error says it:
#                  "Use QuantizationPolicy.fp8_cast() for BF16 checkpoints."
#                  This is the L40S trap -- the capability check passes and the
#                  run still fails, so check_gpu_supports() probes the checkpoint
#                  header as well.
#
#   fp8-cast       stores the covered Linear weights as torch.float8_e4m3fn and
#                  upcasts them back to bf16 inside forward(), then calls a plain
#                  torch.nn.functional.linear (Fp8CastLinear.forward in
#                  fp8_cast.py). There is no _scaled_mm, no Triton on the
#                  inference path (stochastic rounding is off for
#                  UPCAST_DURING_INFERENCE), and no device-capability check
#                  anywhere in the file. FP8 is used purely as a storage dtype.
#                  Upstream documents it as "No extra dependencies".
#
# So fp8-cast is the only quantization reachable on this deployment, and it is
# reachable regardless of GPU generation. The downcast covers to_q/to_k/to_v/
# to_out.0 plus both the video and audio FF projections in every transformer
# block. It does NOT cover the cross-modal add_q/k/v_proj, to_gate_logits, adaln,
# norms, or patchify/proj_out -- those stay bf16 -- so expect the 22B to go from
# ~44 GB to roughly 26-30 GB rather than a clean halving.
#
# Note what fp8-cast does *not* do on an L40S: because it upcasts to bf16 before
# the matmul, it never touches the Ada FP8 tensor cores. The win is memory and
# bandwidth, not arithmetic. The backend that would use those cores is
# fp8-scaled-mm, and it is blocked by checkpoint availability, not by hardware.
#
# The memory win is still the one that matters. Under OffloadMode.CPU the job is
# bound by bytes crossing PCIe per forward pass, and ~28 GB plausibly fits
# resident on a 48 GB L40S -- which removes the streaming entirely and lets Ada's
# considerably better bf16 throughput actually show up.
#
# DEFAULT: fp8-cast + cpu offload.
#
# This is the safe rung of the ladder, not the fast one. Offloading bounds peak
# VRAM at the streaming buffer regardless of how large the weights turn out to
# be, so this combination cannot OOM the way offload=none can -- while still
# cutting roughly a third of the bytes that cross PCIe on every forward pass,
# which is the cost that dominates this endpoint.
#
# Next step once a baseline render looks right:
#
#   LTX_OFFLOAD_MODE=none   <- the target on a 48 GB L40S
#
# That removes streaming entirely and is where Ada's bf16 throughput finally
# shows up. Compare stages_s.generate and peak_vram_gb in the job_timing log
# line. It has headroom at 768x512x49 (stage 2 is only ~2.7k tokens) and gets
# tighter as resolution and frame count grow; if it OOMs, come back here.
#
# Rollback, if fp8-cast misbehaves on this hardware or the quality drop is
# visible: set LTX_QUANTIZATION=none. That is the LTX-2.3 endpoint's proven
# configuration and needs no rebuild -- it is an endpoint env var.
#
# Not reachable here: fp8-scaled-mm (no fp8 checkpoint published), nvfp4-cast /
# nvfp4-prequant (Blackwell SM 10+ and the ltx-kernels package).
DEFAULT_QUANTIZATION = os.getenv("LTX_QUANTIZATION", "fp8-cast")
DEFAULT_OFFLOAD_MODE = os.getenv("LTX_OFFLOAD_MODE", "cpu")
ALLOW_REQUEST_OVERRIDE = os.getenv("ALLOW_REQUEST_PIPELINE_OVERRIDE", "0") == "1"

# Only consulted when the video VAE is the diffusion decoder. CHUNKED_EAGER is
# upstream's default and needs no extra dependencies; the compile modes trade
# a large one-time compile for a faster warm decode, which is a bad deal on a
# scale-to-zero endpoint where workers do not stay warm for long.
DIFFVAE_OPTIMIZATION = os.getenv("LTX_DIFFVAE_OPTIMIZATION", "chunked_eager")

# LTX-2.5 geometry constraints, enforced here so a bad request fails in
# milliseconds instead of after a multi-minute model build. Both come from
# upstream: assert_resolution(is_two_stage=True) requires multiples of 64
# because stage 1 runs at half resolution and must still land on the VAE's 32px
# spatial grid, and the causal VAE needs (frames - 1) % 8 == 0.
RESOLUTION_MULTIPLE = 64
FRAME_TEMPORAL_SCALE = 8

S3_CRED_KEYS = ("endpointUrl", "accessId", "accessSecret")
S3_REQUIRED_KEYS = S3_CRED_KEYS + ("bucketName",)


def resolve_s3_config(job_input):
    """Resolve the destination bucket for this job's video.

    Note on "pass it in the header": RunPod's gateway consumes the HTTP headers
    sent to /run and /runsync -- the handler is only ever given the job id and
    the `input` object. Per-request bucket config therefore has to travel inside
    input.s3_config. This is the same contract the LTX-2.3 worker uses, so no
    client change is needed.

    Endpoint BUCKET_* env vars act as a fallback so a single-bucket deployment
    can keep its secrets out of request payloads entirely.

    For DigitalOcean Spaces, `region` may be given instead of `endpointUrl` and
    the standard Spaces endpoint is derived from it.

    Returns None when neither source is configured. Raises ValueError naming
    only the missing keys, never their values.
    """
    config = dict(job_input.get("s3_config") or {})

    if not config:
        from_env = {
            "endpointUrl": os.getenv("BUCKET_ENDPOINT_URL"),
            "accessId": os.getenv("BUCKET_ACCESS_KEY_ID"),
            "accessSecret": os.getenv("BUCKET_SECRET_ACCESS_KEY"),
            "bucketName": os.getenv("BUCKET_NAME"),
        }
        config = from_env if all(from_env.values()) else {}

    if not config:
        return None

    # DO Spaces convenience: nyc3 -> https://nyc3.digitaloceanspaces.com. Only
    # applied when endpointUrl was not given, so an explicit endpoint (including
    # a CDN or a non-DO S3) always wins.
    if not config.get("endpointUrl") and config.get("region"):
        config["endpointUrl"] = f"https://{config['region']}.digitaloceanspaces.com"

    missing = [key for key in S3_REQUIRED_KEYS if not config.get(key)]
    if missing:
        raise ValueError(f"s3_config is missing required keys: {', '.join(missing)}")

    return config


def upload_video(output_path, job_input):
    """Upload the rendered video and return a presigned URL valid for 7 days."""
    s3_config = resolve_s3_config(job_input)
    if s3_config is None:
        raise ValueError(
            "No bucket configured. Pass input.s3_config with endpointUrl (or "
            "region), accessId, accessSecret and bucketName, or set the BUCKET_* "
            "environment variables on the endpoint."
        )

    return rp_upload.upload_file_to_bucket(
        file_name=f"{uuid.uuid4()}.mp4",
        file_location=output_path,
        bucket_creds={key: s3_config[key] for key in S3_CRED_KEYS},
        bucket_name=s3_config["bucketName"],
        prefix=job_input.get("s3_prefix"),
    )


def silence_use_fast_deprecation():
    """Stop the transformers `use_fast` notice repeating on every job.

    ltx-core builds the Gemma text encoder through AutoImageProcessor with
    use_fast=False, and recent transformers deprecates that argument. The call
    site sits in a dependency pinned by LTX_REF, so nothing here can pass the new
    argument instead -- the fix is upstream's and lands when LTX_REF moves. Until
    then this is purely cosmetic: the notice costs nothing but log noise.

    Filtered by message rather than by lowering transformers' log level, so
    genuine warnings from the library still reach the logs. Both channels are
    covered because transformers routes deprecations through warnings.warn in
    some versions and its own logger in others.
    """
    needle = "`use_fast` parameter is deprecated"

    warnings.filterwarnings("ignore", message=f".*{re.escape(needle)}.*")

    class _DropNotice(logging.Filter):
        def filter(self, record):
            return needle not in record.getMessage()

    try:
        from transformers.utils import logging as hf_logging

        # Force transformers to install its handler before we filter it;
        # it is created lazily on first get_logger() call.
        hf_logging.get_logger()
    except ImportError:
        return

    hf_root = logging.getLogger("transformers")
    hf_root.addFilter(_DropNotice())
    for handler in hf_root.handlers:
        handler.addFilter(_DropNotice())


def redact_url(url):
    """Strip the query string from a URL before logging it.

    Conditioning images arrive as presigned URLs, so the query string carries
    the access key id and signature. Printing it whole puts live credentials
    into RunPod's log storage, where they outlive the job by a long way.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    redacted = "?<redacted>" if parts.query else ""
    return f"{parts.scheme}://{parts.netloc}{parts.path}{redacted}"


def download_file(url, target_path):
    print(f"Downloading input file from {redact_url(url)} to {target_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
        out_file.write(response.read())
    return target_path


class StageTimer:
    """Collect per-stage wall-clock timings for one job and emit them as JSON.

    The endpoint's cost is dominated by work that is invisible in the response:
    the per-call model rebuilds run far longer than the denoising they feed.
    Without per-stage numbers there is no way to tell a config change that
    helped from one that did not, so every job reports where its seconds went
    and how close it came to filling VRAM.
    """

    def __init__(self, job_id):
        self.job_id = job_id
        self.stages = {}
        self._started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stage(self, name):
        return _StageContext(self, name)

    def record(self, name, seconds):
        self.stages[name] = round(seconds, 2)

    def emit(self, **extra):
        payload = {
            "event": "job_timing",
            "job_id": self.job_id,
            "model": "ltx-2.5-distilled",
            "total_s": round(time.perf_counter() - self._started, 2),
            "stages_s": self.stages,
            "quantization": DEFAULT_QUANTIZATION,
            "offload_mode": DEFAULT_OFFLOAD_MODE,
            "video_vae": VIDEO_VAE_VARIANT,
            "weights_staged_locally": sorted(
                os.path.basename(source)
                for source, target in STAGED_PATHS.items()
                if source != target
            ),
            **extra,
        }
        if torch.cuda.is_available():
            payload["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        print(json.dumps(payload), flush=True)


class _StageContext:
    def __init__(self, timer, name):
        self._timer = timer
        self._name = name

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_):
        # Record on failure too -- knowing a job died 300s into generation is
        # more useful than knowing only that it died. Returning False keeps the
        # exception propagating; this only observes it.
        self._timer.record(self._name, time.perf_counter() - self._started)
        return False


def resolve_pipeline_config(job_input):
    """Return (quantization, offload_mode) for this job.

    Reads the endpoint environment by default so a single request cannot force
    a multi-minute pipeline rebuild on everyone else. See the notes on the
    DEFAULT_* constants for why these belong to the deployment.
    """
    if ALLOW_REQUEST_OVERRIDE:
        quantization = job_input.get("quantization", DEFAULT_QUANTIZATION)
        offload_mode = job_input.get("offload_mode", DEFAULT_OFFLOAD_MODE)
    else:
        quantization, offload_mode = DEFAULT_QUANTIZATION, DEFAULT_OFFLOAD_MODE

    return (None if quantization in ("none", "", None) else quantization), offload_mode


def stage_weights():
    """Copy the weights onto local disk, highest-value file first.

    See model_paths_config.staging_candidates() for the ordering rationale.
    Anything that does not fit is skipped and served from the network volume,
    so this is safe to leave on regardless of container disk size.
    """
    global STAGED_PATHS

    candidates = staging_candidates()
    if not STAGING_ENABLED:
        print("[staging] disabled via DISABLE_LOCAL_STAGING")
        STAGED_PATHS = {path: path for path in candidates}
        return

    started = time.perf_counter()
    STAGED_PATHS = stage_files(candidates, LOCAL_CACHE_DIR, reserve_bytes=STAGING_RESERVE_BYTES)
    staged = [os.path.basename(p) for p, t in STAGED_PATHS.items() if p != t]
    print(
        f"[staging] done in {time.perf_counter() - started:.0f}s; "
        f"local: {staged or 'none (serving from network volume)'}"
    )


def resolve_weight_path(path):
    if path is None:
        return None
    return STAGED_PATHS.get(path, path)


def build_model_paths():
    """Assemble the split-pack component paths, resolved through the staging map.

    ModelPaths.from_split leaves omitted slots as None and raises at the use
    site rather than filling them with the transformer -- which is what makes
    a missing audio VAE a clear error instead of an uninitialised model that
    silently assumes default 32x32x8 scale factors.
    """
    return ModelPaths.from_split(
        transformer_path=resolve_weight_path(TRANSFORMER_PATH),
        text_encoder_path=resolve_weight_path(TEXT_ENCODER_PATH),
        video_vae_path=resolve_weight_path(VIDEO_VAE_PATH),
        audio_vae_path=resolve_weight_path(AUDIO_VAE_PATH),
        duration_head_path=resolve_weight_path(DURATION_HEAD_PATH),
    )


# Minimum CUDA compute capability each backend needs, and why. Checked at
# startup so a misconfigured endpoint fails on boot with a readable message
# instead of somewhere inside a forward pass, minutes into a paid job.
_QUANT_MIN_CAPABILITY = {
    # torch._scaled_mm needs native FP8 tensor cores (Ada SM 8.9+).
    "fp8-scaled-mm": (8, 9),
    # ltx_kernels.nvfp4 requires Blackwell. Capability alone is not sufficient --
    # see _QUANT_REQUIRES_LTX_KERNELS below.
    "nvfp4-cast": (10, 0),
    "nvfp4-prequant": (10, 0),
    # fp8-cast deliberately absent: it uses FP8 only as a storage dtype and
    # upcasts to bf16 before a plain F.linear, so it has no capability floor
    # beyond what PyTorch needs to hold a float8_e4m3fn tensor.
}

# Backends that additionally need the compiled ltx-kernels extension, which this
# image does not install (it builds CUDA sources and wants a matching nvcc).
# Capability is not enough: a workstation Blackwell such as the RTX PRO 6000 is
# sm_120, which clears the SM >= 10.0 floor and would sail past the capability
# check straight into an ImportError on ltx_kernels.nvfp4.
_QUANT_REQUIRES_LTX_KERNELS = frozenset({"nvfp4-cast", "nvfp4-prequant"})


def _ltx_kernels_available():
    """True when the compiled nvfp4 extension can actually be imported."""
    try:
        from ltx_kernels import nvfp4  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


def check_torch_supports_arch(capability):
    """Warn when the installed torch has no kernels for this GPU's architecture.

    The failure this prevents is one of the most confusing in CUDA: a torch build
    without SASS for the device still imports, still reports the GPU, still
    allocates -- and then dies on the first real kernel with "no kernel image is
    available for execution on the device", or silently falls back to a PTX JIT
    that is slow enough to look like a hang.

    Blackwell is where this bites now. A workstation RTX PRO 6000 is sm_120 and
    needs a CUDA 12.8+ torch build; a cu126 wheel tops out below that and will
    not carry sm_120 SASS. Comparing the device against torch.cuda.get_arch_list()
    turns a mid-generation crash into one line at boot.
    """
    device_arch = f"sm_{capability[0]}{capability[1]}"
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return

    print(f"[gpu] torch {torch.__version__} built for: {' '.join(arch_list) or '<none>'}")
    if not arch_list or device_arch in arch_list:
        return

    # PTX for an older arch of the same major family can JIT forward-compatibly,
    # but it is a performance cliff rather than a fix -- still worth flagging.
    print(
        f"[gpu] WARNING: this torch build has no {device_arch} kernels. Expect "
        f"'no kernel image is available for execution on the device', or a very slow "
        f"PTX JIT fallback. Rebuild the image with a torch wheel that targets "
        f"{device_arch} -- for Blackwell that means the CUDA 12.8+ index, e.g. "
        f"pip install torch --index-url https://download.pytorch.org/whl/cu128"
    )


def check_gpu_supports(quantization_str):
    """Warn loudly when the configured quantization cannot run here.

    Two independent reasons it might not, and they fail in different places:

      capability  -- the GPU lacks the tensor cores the backend needs. Surfaces
                     as a CUDA error during the first denoising step.
      checkpoint  -- fp8-scaled-mm additionally requires a *pre-quantized* fp8
                     checkpoint (F8_E4M3 .weight plus a sibling .weight_scale).
                     LTX-2.5 publishes bf16 transformers only, so this raises
                     from get_fp8_swap_module_ops at pipeline construction even
                     on hardware that fully supports FP8.

    The second is the L40S trap: SM 8.9 passes the capability check, so without
    the checkpoint probe the operator would reasonably conclude fp8-scaled-mm
    should work and be confused by the failure.

    Deliberately warnings rather than hard failures: this is the worker's belief
    about upstream, not upstream's own gate, and being wrong in the conservative
    direction should not brick an endpoint.
    """
    if not torch.cuda.is_available():
        print("[gpu] CUDA not available -- running on CPU, expect this to be unusably slow")
        return

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[gpu] {name}, compute capability {capability[0]}.{capability[1]}, {total_gb:.0f} GB")

    check_torch_supports_arch(capability)

    if not quantization_str:
        return

    if quantization_str in _QUANT_REQUIRES_LTX_KERNELS and not _ltx_kernels_available():
        print(
            f"[gpu] WARNING: LTX_QUANTIZATION={quantization_str} needs the compiled "
            f"ltx-kernels extension, which this image does not install. The GPU may well "
            f"be capable -- a workstation Blackwell (RTX PRO 6000) is sm_120 and clears "
            f"the SM >= 10.0 floor -- but ltx_kernels.nvfp4 will fail to import. Add "
            f"ltx-kernels to the Dockerfile with a matching nvcc and a TORCH_CUDA_ARCH_LIST "
            f"covering your card, or use 'fp8-cast'."
        )

    required = _QUANT_MIN_CAPABILITY.get(quantization_str)
    if required and capability < required:
        print(
            f"[gpu] WARNING: LTX_QUANTIZATION={quantization_str} needs compute "
            f"capability >= {required[0]}.{required[1]} but this GPU is "
            f"{capability[0]}.{capability[1]}. Use 'fp8-cast' (stores weights in FP8, "
            f"upcasts to bf16 for a plain matmul -- no capability floor) or 'none'."
        )

    if quantization_str == "fp8-scaled-mm" and not checkpoint_has_fp8_weights(TRANSFORMER_PATH):
        print(
            "[gpu] WARNING: LTX_QUANTIZATION=fp8-scaled-mm requires a pre-quantized fp8 "
            "checkpoint (F8_E4M3 .weight + .weight_scale). This deployment downloads "
            f"{os.path.basename(TRANSFORMER_PATH)}, which is bf16, so pipeline "
            "construction will raise. Use 'fp8-cast' instead -- it downcasts a bf16 "
            "checkpoint on the fly and is the supported path for the weights we ship."
        )


def checkpoint_has_fp8_weights(path):
    """True when the safetensors file carries prequantized fp8 weight/scale pairs.

    Reads only the header, so this is cheap even on a 44 GB file. Returns True on
    any error so a probe failure never blocks a startup that might otherwise work.
    """
    if not path or not os.path.exists(path):
        return True  # not downloaded yet; nothing to assert
    try:
        import safetensors  # noqa: PLC0415

        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            return any(key.endswith(".weight_scale") for key in handle.keys())
    except Exception as error:
        print(f"[gpu] could not inspect {os.path.basename(path)}: {error}")
        return True


def get_pipeline(quantization_str, offload_str):
    global current_pipeline, current_quantization, current_offload_mode

    if (
        current_pipeline is not None
        and current_quantization == quantization_str
        and current_offload_mode == offload_str
    ):
        return current_pipeline

    if current_pipeline is not None:
        print("Cleaning up old pipeline instance...")
        del current_pipeline
        current_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Initializing LTX-2.5 DistilledPipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = build_model_paths()

    # The quantization policy is derived from the transformer checkpoint, since
    # that is the only component this deployment quantizes. Passing the wrong
    # component here is a silent misconfiguration -- fp8-scaled-mm and
    # nvfp4-prequant read scale tensors out of the file they are handed.
    quantization_kind = QuantizationKind(quantization_str) if quantization_str else None
    quantization_policy = (
        quantization_kind.to_policy(model_paths.transformer()) if quantization_kind else None
    )

    current_pipeline = DistilledPipeline(
        model_paths=model_paths,
        spatial_upsampler_path=resolve_weight_path(SPATIAL_UPSAMPLER_PATH),
        loras=[],
        device=device,
        quantization=quantization_policy,
        offload_mode=OffloadMode(offload_str) if offload_str else OffloadMode.NONE,
        diffvae_optimization=DiffVAEMode(DIFFVAE_OPTIMIZATION),
    )

    current_quantization = quantization_str
    current_offload_mode = offload_str
    return current_pipeline


def validate_geometry(height, width, num_frames):
    """Reject unrenderable dimensions before any model is built.

    Upstream checks the same things, but only after prompt encoding -- which on
    this endpoint means several minutes of Gemma load before a typo in `width`
    surfaces. Both rules are properties of the VAE geometry, not preferences.
    """
    errors = []
    if not isinstance(num_frames, int):
        num_frames = None  # AutoDuration sentinel: the pipeline snaps it to the grid itself
    if height % RESOLUTION_MULTIPLE or width % RESOLUTION_MULTIPLE:
        errors.append(
            f"height and width must both be multiples of {RESOLUTION_MULTIPLE} for the "
            f"two-stage distilled pipeline (stage 1 renders at half resolution and must "
            f"still land on the VAE's 32px grid); got {width}x{height}"
        )
    if num_frames is not None and (num_frames - 1) % FRAME_TEMPORAL_SCALE:
        nearest = ((num_frames - 1) // FRAME_TEMPORAL_SCALE) * FRAME_TEMPORAL_SCALE + 1
        errors.append(
            f"num_frames must satisfy (num_frames - 1) % {FRAME_TEMPORAL_SCALE} == 0 "
            f"(causal VAE temporal grid); got {num_frames}, nearest valid is {nearest}"
        )
    if errors:
        raise ValueError("; ".join(errors))


def resolve_num_frames_request(job_input):
    """Read num_frames, or request auto-duration when the duration head is present.

    Omitting num_frames only works when the duration-head checkpoint was
    downloaded (LTX_ENABLE_DURATION_HEAD=1, the default). Without it upstream
    raises from require_num_frames_source, so the default is filled in here to
    keep that a config choice rather than a runtime surprise.
    """
    if "num_frames" in job_input and job_input["num_frames"] is not None:
        return int(job_input["num_frames"])
    if DURATION_HEAD_PATH is not None and job_input.get("auto_duration"):
        # Sentinel the pipeline understands: predict the frame count from the
        # caption via DurationHead, snapped to the VAE's temporal grid.
        return DEFAULT_AUTO_DURATION
    return 49  # ~2s at 24fps; short default keeps dev cycles fast


# Upstream guards its CLI entrypoints with this, not the pipeline classes, so
# calling DistilledPipeline directly leaves autograd on. The weights load as
# inference tensors, autograd then tries to save them for a backward pass that
# will never happen, and the first F.linear of the first denoising step dies
# with "Inference tensors cannot be saved for backward". Covers construction
# as well as the call, exactly as upstream's main() does.
@torch.inference_mode()
def handler(event):
    job_input = event.get("input", {})
    if not job_input:
        return {"error": "No input configuration provided."}

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A prompt must be provided."}

    # negative_prompt is intentionally not read: the distilled pipeline runs
    # guidance-free on its fixed sigma schedule (8 steps stage 1, 3 stage 2) and
    # has no negative-prompt input. Same for num_inference_steps and the
    # cfg/stg knobs -- those belong to the two-stage full-model pipelines.
    seed = int(job_input.get("seed", 42))
    height = int(job_input.get("height", 512))
    width = int(job_input.get("width", 768))
    frame_rate = float(job_input.get("frame_rate", 25.0))
    num_frames = resolve_num_frames_request(job_input)

    quantization, offload_mode = resolve_pipeline_config(job_input)
    image_conditioning = job_input.get("image_conditioning", [])
    temp_files = []
    timer = StageTimer(event.get("id"))

    try:
        validate_geometry(height, width, num_frames)

        with timer.stage("pipeline_init"):
            pipe = get_pipeline(quantization_str=quantization, offload_str=offload_mode)

        with timer.stage("input_download"):
            images_input = []
            for idx, img_cond in enumerate(image_conditioning):
                url = img_cond.get("url")
                if not url:
                    continue
                local_path = f"/tmp/{uuid.uuid4()}_{idx}.png"
                download_file(url, local_path)
                temp_files.append(local_path)
                # crf is left unset on purpose. ImageConditioner.resolve_crf
                # fills it from the checkpoint's model_version, so conditioning
                # images are re-compressed at whatever LTX-2.5 was trained with
                # rather than at 2.3's value. Pass crf explicitly (0 = no
                # re-compression) only to override that.
                images_input.append(
                    ImageConditioningInput(
                        path=local_path,
                        frame_idx=int(img_cond.get("frame_idx", 0)),
                        strength=float(img_cond.get("strength", 1.0)),
                        crf=img_cond.get("crf"),
                    )
                )

        # AUTO_TILING lets the pipeline derive the tile layout from the VAE's
        # real compression factors and the requested shape. The resolved config
        # comes back out of the call, which is the only way to compute the
        # chunk count encode_video needs without duplicating that logic.
        with timer.stage("generate"):
            video_gen, audio_gen, resolved_frames, tiling_config = pipe(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images_input,
                tiling_config=AUTO_TILING,
            )

        output_filename = f"/tmp/{uuid.uuid4()}.mp4"
        temp_files.append(output_filename)

        with timer.stage("encode"):
            encode_video(
                video=video_gen,
                fps=frame_rate,
                audio=audio_gen,
                output_path=output_filename,
                video_chunks_number=get_video_chunks_number(resolved_frames, tiling_config),
            )

        with timer.stage("upload"):
            video_url = upload_video(output_filename, job_input)

        timer.emit(
            outcome="success",
            height=height,
            width=width,
            num_frames=resolved_frames,
        )
        return {
            "status": "success",
            "video_url": video_url,
            "seed": seed,
            "num_frames": resolved_frames,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
        }

    except Exception as e:
        import traceback

        # Log the traceback rather than only returning it: the job record is
        # not somewhere you can grep, and a failure that leaves no trace in the
        # container logs is a failure you cannot debug after the fact.
        traceback.print_exc()
        timer.emit(outcome="failed", error_type=type(e).__name__)
        return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def boot():
    """Everything that must happen once per worker, before accepting traffic.

    Split out of the __main__ block so rp_handler.py -- the entrypoint RunPod's
    Dockerfile actually runs -- can call it without duplicating the sequence.
    Importing this module never triggers any of it.
    """
    # Identity banner, first line of every worker's log. Both this image and the
    # LTX-2.3 worker derive from pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel, so
    # the NVIDIA CUDA 12.1.1 header the base image prints is identical in both
    # and cannot be used to tell them apart. Without this line the first
    # distinguishing output is a [models] message most of the way into startup.
    print(
        f"[worker] ltx-2.5-distilled | quantization={DEFAULT_QUANTIZATION} "
        f"offload={DEFAULT_OFFLOAD_MODE} video_vae={VIDEO_VAE_VARIANT} "
        f"models_root={MODELS_ROOT}",
        flush=True,
    )

    silence_use_fast_deprecation()

    # Populate the model weights before serving. On a mounted network volume
    # (MODELS_ROOT=/runpod-volume/models) this downloads once and every later
    # cold start reuses the cached files instead of re-pulling ~66 GiB.
    ensure_models(MODELS_ROOT)

    # Report the GPU and sanity-check the configured quantization. Runs after the
    # download because one of the two checks has to read the transformer's
    # safetensors header -- nothing is wasted, since the weights are needed
    # whatever the quantization setting turns out to be.
    check_gpu_supports(resolve_pipeline_config({})[0])

    # Then copy them onto local disk, because the network volume is the
    # bottleneck rather than the GPU. Paid once per worker, and FlashBoot
    # amortises it across every job that worker goes on to serve.
    stage_weights()

    # Construct the pipeline before accepting traffic. The object graph is
    # cheap to build (upstream defers the expensive weight loads to call time),
    # but doing it here keeps it off the first request's clock and turns a bad
    # checkpoint path into a startup failure instead of a failed job.
    get_pipeline(*resolve_pipeline_config({}))


if __name__ == "__main__":
    boot()
    runpod.serverless.start({"handler": handler})
