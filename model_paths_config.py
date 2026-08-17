"""Component layout for the LTX-2.5 split pack.

Single source of truth for which files this worker needs and where they live,
shared by ``download_models.py`` (fetches them), ``handler.py`` (loads them) and
``staging.py`` (copies them onto local disk in priority order).

LTX-2.5 is a *split* pack. The 2.3 worker loaded one fat checkpoint that carried
the transformer, both VAEs and the text-encoder projections together, plus a
Gemma-3 directory. Here every component is its own safetensors file and Gemma
ships pre-packed with the LTX text projections, so ``ModelPaths.from_split`` is
the only correct constructor -- ``from_monolith`` would point every slot at the
transformer, which carries no VAE config or weights at all.
"""

import os

MODELS_ROOT = os.getenv("MODELS_ROOT", "/workspace/models")
LTX_DIR = os.path.join(MODELS_ROOT, "ltx-2.5")

HF_REPO = "Lightricks/LTX-2.5"

# The Hugging Face CLI preserves the repo's folder layout under --local-dir, and
# hf_hub_download does the same, so these keep their subdirectory prefixes.
TRANSFORMER_FILE = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER_FILE = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
AUDIO_VAE_FILE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSAMPLER_FILE = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

# Two video VAEs ship, and the choice is a real deployment decision rather than a
# preference:
#
#   conv      -- convolutional decoder. Lighter, faster, no extra dependencies.
#   diffusion -- NADiffusionDecoder. Better detail, but a longer decode and more
#                VRAM, and it is only fast with the `natten` extra, which pins
#                torch==2.13.0+cu132 and is Linux/CUDA only. Without natten it
#                falls back to Triton or eager neighborhood attention, which is
#                the slowest path of the three.
#
# Serverless bills per second and the decode sits on the critical path of every
# job, so conv is the default. Set LTX_VIDEO_VAE=diffusion to opt in, and expect
# to add the natten extra to the image before it pays for itself.
VIDEO_VAE_VARIANT = os.getenv("LTX_VIDEO_VAE", "conv")
_VIDEO_VAE_FILES = {
    "conv": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "diffusion": "vae/ltx-2.5-video-vae-bf16.safetensors",
}
if VIDEO_VAE_VARIANT not in _VIDEO_VAE_FILES:
    raise ValueError(
        f"LTX_VIDEO_VAE={VIDEO_VAE_VARIANT!r} is not valid. "
        f"Expected one of: {', '.join(sorted(_VIDEO_VAE_FILES))}."
    )
VIDEO_VAE_FILE = _VIDEO_VAE_FILES[VIDEO_VAE_VARIANT]

# Optional, a few MB. Predicts a frame count from the caption so callers may omit
# num_frames. DistilledPipeline treats a missing duration head as "num_frames is
# mandatory" rather than an error (DurationPredictor.from_checkpoint returns None
# and require_num_frames_source raises only if you then omit num_frames), so
# skipping it costs nothing as long as every request carries num_frames.
DURATION_HEAD_FILE = "model_patches/ltx-2.5-duration-head-bf16.safetensors"
ENABLE_DURATION_HEAD = os.getenv("LTX_ENABLE_DURATION_HEAD", "1") != "0"


def _local(relative_file: str) -> str:
    return os.path.join(LTX_DIR, relative_file)


TRANSFORMER_PATH = _local(TRANSFORMER_FILE)
TEXT_ENCODER_PATH = _local(TEXT_ENCODER_FILE)
VIDEO_VAE_PATH = _local(VIDEO_VAE_FILE)
AUDIO_VAE_PATH = _local(AUDIO_VAE_FILE)
SPATIAL_UPSAMPLER_PATH = _local(SPATIAL_UPSAMPLER_FILE)
DURATION_HEAD_PATH = _local(DURATION_HEAD_FILE) if ENABLE_DURATION_HEAD else None


def required_repo_files() -> list[str]:
    """Repo-relative files this deployment downloads, largest-first.

    Ordered so an interrupted download makes progress on the files that dominate
    the total rather than finishing the small ones first.
    """
    files = [
        TRANSFORMER_FILE,
        TEXT_ENCODER_FILE,
        VIDEO_VAE_FILE,
        SPATIAL_UPSAMPLER_FILE,
        AUDIO_VAE_FILE,
    ]
    if ENABLE_DURATION_HEAD:
        files.append(DURATION_HEAD_FILE)
    return files


def staging_candidates() -> list[str]:
    """Absolute paths to stage onto local disk, highest value first.

    Priority is *bytes read per job*, not size on disk. Upstream frees every
    component after use and rebuilds it from its checkpoint on the next call, so
    a file's cost is its size times how often the pipeline reopens it:

      transformer       ~44 GB x 2  (DiffusionStage runs for stage 1 and stage 2)
      text encoder      ~24 GB x 1  (PromptEncoder, once per job)
      video VAE          x 4        (ImageConditioner twice, VideoUpsampler,
                                     VideoDecoder -- one file, four opens)
      spatial upsampler  ~1 GB x 1
      audio VAE          small x 1
      duration head      few MB x 0 (held resident; never re-read)

    stage_files skips anything that will not fit and leaves it pointing at the
    network volume, so listing more than the disk can hold is safe.
    """
    candidates = [
        TRANSFORMER_PATH,
        TEXT_ENCODER_PATH,
        VIDEO_VAE_PATH,
        SPATIAL_UPSAMPLER_PATH,
        AUDIO_VAE_PATH,
    ]
    if DURATION_HEAD_PATH:
        candidates.append(DURATION_HEAD_PATH)
    return candidates
