# LTX-2.5 — RunPod Serverless Worker

Text/image-to-video with synchronized audio, using **LTX-2.5 `DistilledPipeline`**.
Renders to mp4 and uploads to a DigitalOcean Spaces bucket supplied per request.

Successor to `runpod-ltx2-worker` (LTX-2.3). **The request contract is unchanged**, so
existing callers work against this endpoint without modification.

---

## Why `DistilledPipeline`

The monorepo ships 15 pipelines. Only three run on the distilled transformer alone —
`DistilledPipeline`, `ICLoraPipeline` (video-to-video, needs an IC-LoRA file and an input
video) and `DubItPipeline` (lip-dub). Every other quality pipeline — `TI2VidTwoStages`,
`TI2VidTwoStagesHQ`, `A2Vid`, `KeyframeInterpolation`, `DFR` — needs the **dev**
transformer *plus* the distilled LoRA: a second ~44 GB checkpoint to download, stage and
re-read on every job, for 30 denoising steps instead of 8 + 3.

On a scale-to-zero endpoint billed per second, that is the whole decision.

What LTX-2.5 adds over the 2.3 deployment, at no configuration cost:

| | |
|---|---|
| **Ancestral stage-1 sampler** | `EulerAncestralDiffusionStep(eta=1.0)` replaces plain Euler on stage 1. Enabled automatically — the pipeline reads `model_version` from the checkpoint (`detect_model_version() >= (2,5)`). Nothing to pass. |
| **Auto duration** | Optional `DurationHead` predicts frame count from the caption. Send `"auto_duration": true` and omit `num_frames`. |
| **Checkpoint-driven CRF** | Conditioning images are re-compressed at whatever CRF LTX-2.5 was trained with (18), not 2.3's value (33). Handled by `ImageConditioner.resolve_crf`. |
| **NVFP4** | Available on Blackwell via `LTX_QUANTIZATION=nvfp4-cast` — requires adding `ltx-kernels` to the image first. |

---

## Models

LTX-2.5 is a **split pack**. In 2.3 the distilled checkpoint was a monolith bundling the
transformer, both VAEs and the text-encoder projections; in 2.5 each component is its own
file, and `ltx-2.5-22b-distilled-transformer-bf16.safetensors` is the transformer **only**.

`DistilledPipeline.__init__` unconditionally builds a `PromptEncoder`, `ImageConditioner`,
`VideoUpsampler`, `VideoDecoder` and `AudioDecoder`, so five files are mandatory:

| Component | File (in `Lightricks/LTX-2.5`) | Required |
|---|---|---|
| Transformer | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | yes |
| Text encoder | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | yes |
| Video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | yes |
| Audio VAE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | yes |
| Spatial upsampler | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | yes |
| Duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | optional (few MB) |

Roughly 66 GiB total. The text encoder is the LTX-tuned Gemma 4 with projections already
packed in — **Google's stock Gemma 4 is not a substitute**; loading validates the
encoder against the checkpoint's `gemma_source_checkpoint` metadata and rejects a mismatch.
This also means there is no second Gemma download, unlike the 2.3 worker.

`Lightricks/LTX-2.5` is **gated**. Set `HF_TOKEN` to a Read token from an account that has
accepted the model terms, or the first boot fails with 401/403.

### Video VAE: conv vs diffusion

Two decoders ship and the choice is a real deployment decision, not a preference.

- **`conv`** (default) — convolutional. Lighter, faster, no extra dependencies.
- **`diffusion`** — `NADiffusionDecoder`. Better detail, but longer decode and more VRAM,
  and it is only fast with the `natten` extra, which pins `torch==2.13.0+cu132` and would
  fight this image's torch pins. Without natten it falls back to Triton or eager
  neighborhood attention — the slowest of the three paths.

Decode sits on the critical path of every job, so `LTX_VIDEO_VAE=conv` is the default.
Switching to `diffusion` also changes the download set, which is why the volume's
completion marker is keyed on the variant.

---

## Deployment

### 1. Build

CI (`.github/workflows/build.yml`) pushes to GHCR on manual dispatch. Or locally:

```bash
docker build -t your-registry/ltx25-worker:latest .
docker push your-registry/ltx25-worker:latest
```

The build ends with an API-surface check that imports every LTX symbol the handler uses
and asserts `DistilledPipeline.__init__` still takes `model_paths`. A wrong `LTX_REF`
fails the build instead of a cold start after RunPod has pulled a 20 GB image.

### 2. Endpoint configuration

| Setting | Value |
|---|---|
| GPU | **L40S (48 GB)** — see the tuning section below |
| Container disk | **100 GB+** — holds the staged weight copies (~66 GiB) |
| Network volume | **150 GB+**, mounted at `/runpod-volume` |
| `MODELS_ROOT` | `/runpod-volume/models` (so weights survive worker churn) |
| Max workers | as needed; **min workers 0** is fine |
| FlashBoot | on — it is what amortises the staging cost |

### 3. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | **Required.** Read token with LTX-2.5 terms accepted. |
| `MODELS_ROOT` | `/workspace/models` | Set to `/runpod-volume/models` in production. |
| `LTX_QUANTIZATION` | `fp8-cast` | `none`, `fp8-cast`. **Not** `fp8-scaled-mm` or `nvfp4-*` — see below. |
| `LTX_OFFLOAD_MODE` | `cpu` | `none`, `cpu`, `disk`. Try `none` after a good baseline render. |
| `LTX_VIDEO_VAE` | `conv` | `conv` or `diffusion`. |
| `LTX_ENABLE_DURATION_HEAD` | `1` | `0` skips the duration-head download. |
| `LTX_DIFFVAE_OPTIMIZATION` | `chunked_eager` | Only used when `LTX_VIDEO_VAE=diffusion`. |
| `DISABLE_LOCAL_STAGING` | `0` | `1` serves weights straight off the network volume. |
| `STAGING_RESERVE_GB` | `5` | Headroom left on the container disk after staging. |
| `ALLOW_REQUEST_PIPELINE_OVERRIDE` | `0` | `1` lets a request set quantization/offload. See below. |
| `BUCKET_*` | — | Optional single-bucket fallback (see *Storage*). |

### Tuning for the L40S — read this before your first production run

The L40S is Ada Lovelace, SM 8.9, 48 GB. The image ships with
**`LTX_QUANTIZATION=fp8-cast`, `LTX_OFFLOAD_MODE=cpu`** — the safe rung of the ladder, not
the fast one. Offloading bounds peak VRAM at the streaming buffer, so this pair cannot OOM,
while still cutting roughly a third of the bytes crossing PCIe per forward pass.

**Rollback** if `fp8-cast` misbehaves on your hardware or the quality drop is visible: set
`LTX_QUANTIZATION=none` on the endpoint. That is the LTX-2.3 endpoint's proven
configuration and needs no rebuild.

#### `fp8-scaled-mm` is the trap. Do not reach for it.

SM 8.9 is exactly where FP8 tensor cores arrive, so the obvious move looks like
`fp8-scaled-mm`. **It will fail**, and not for a hardware reason:

```
fp8_scaled_mm requires a pre-quantized checkpoint with F8_E4M3 .weight +
.weight_scale tensors, but '...-bf16.safetensors' has none.
Use QuantizationPolicy.fp8_cast() for BF16 checkpoints.
```

`get_fp8_swap_module_ops` raises at pipeline construction unless the checkpoint ships fp8
weights with sibling scale tensors. LTX-2.5 publishes **bf16 transformers only** — the
README lists exactly two, both `-bf16`. LTX-2.3 had fp8 variants; 2.5 does not.

So the capability check passes and the run still fails. The worker's startup check probes
the checkpoint header for this specific case and warns before you hit it.

#### `fp8-cast` is the one that works — but understand what it buys

| Backend | Mechanism | L40S? |
|---|---|---|
| `fp8-scaled-mm` | `torch._scaled_mm` (`fp8_scaled_mm.py:59`). Uses Ada's FP8 cores. **Needs a pre-quantized fp8 checkpoint.** | **No** — hardware qualifies, checkpoint does not exist. |
| `fp8-cast` | Stores covered Linear weights as `torch.float8_e4m3fn`, **upcasts to bf16 inside `forward()`**, then a plain `F.linear` (`Fp8CastLinear.forward`). No `_scaled_mm`, no Triton on the inference path, no capability check anywhere in the file. | **Yes.** FP8 as a storage dtype only. |

Note the consequence: **`fp8-cast` never touches the L40S's FP8 tensor cores.** It upcasts
before the matmul. The win is memory and bandwidth, not arithmetic. The backend that would
use those cores is blocked by checkpoint availability.

The memory win is still the one that matters here. Covered: `to_q`/`to_k`/`to_v`/`to_out.0`
plus both video and audio FF projections in every block. Not covered — these stay bf16:
cross-modal `add_q/k/v_proj`, `to_gate_logits`, adaln, norms, patchify/`proj_out`. Expect
**~44 GB → roughly 26–30 GB**, not a clean halving.

#### Test ladder

Change one env var, re-run `test_input.json`, compare `stages_s.generate` and
`peak_vram_gb` in the `job_timing` log line:

| Step | `LTX_QUANTIZATION` | `LTX_OFFLOAD_MODE` | Expectation |
|---|---|---|---|
| **1 — ships as default** | `fp8-cast` | `cpu` | Cannot OOM; VRAM bounded by the streaming buffer. Roughly a third less PCIe traffic per pass. |
| **2 — try next** | `fp8-cast` | `none` | **The target on 48 GB.** Removes streaming entirely, which is where Ada's much better bf16 throughput finally shows up. OOMs if ~28 GB + activations don't fit — fall back to step 1. |
| rollback | `none` | `cpu` | The LTX-2.3 endpoint's configuration. Slowest, but known-good on real hardware. |

Step 2 has real headroom at 768×512×49 (stage 2 is only ~2,700 tokens) and gets tighter as
resolution and frame count grow — re-test it if you move to 1080p or 121+ frames.

**Caveat, stated plainly:** this is read from the implementation, not measured on hardware
— I have no GPU here. Step 1 is low-risk; step 2 either fits or raises a clean CUDA OOM at
startup. `fp8-cast` is also lossy (e4m3, 3 mantissa bits, deterministic rounding on the
inference path), so eyeball a render against the `none` baseline before committing.

#### Everything else L40S-specific

- **Attention:** Ada gets PyTorch SDPA. FlashAttention 3 is Hopper-only (SM 9.0) and FA4 is
  Blackwell-only — neither is installable on an L40S, and there is nothing to configure.
- **Video VAE:** keep `LTX_VIDEO_VAE=conv`. The diffusion decoder without `natten` falls
  back to Triton or eager neighborhood attention — the slowest of the three paths — and
  `natten` would pin `torch==2.13.0+cu132` against this image's pins.
- **CPU RAM:** `OffloadMode.CPU` *pins* the weights in host RAM (~44 GB at bf16). Moot once
  you reach step 2, which is another argument for going there.
- **vs the A40 you were on:** same 48 GB, so the VRAM arithmetic is unchanged and the same
  ladder applies. What you gain is Ada's substantially better bf16 throughput — which only
  becomes visible once step 2 removes the streaming bottleneck that currently dominates.
- **Startup check:** the worker logs GPU name, compute capability and VRAM on boot, warns
  if `LTX_QUANTIZATION` outruns the card, and separately warns if `fp8-scaled-mm` is set
  against a bf16 checkpoint.

`ALLOW_REQUEST_PIPELINE_OVERRIDE` is off by default on purpose: changing either value
rebuilds the pipeline, so honouring them per request lets one caller evict the warm
pipeline for everyone queued behind it. Turn it on only for a deliberate A/B.

---

## API

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run      # async
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync  # blocking
Authorization: Bearer $RUNPOD_API_KEY
```

### Storage credentials go in the body, not the header

RunPod's gateway consumes the HTTP headers sent to `/run` and `/runsync` — the handler is
only ever handed the job id and the `input` object. Bucket credentials therefore travel in
`input.s3_config`. This is the same contract the 2.3 worker uses.

```json
{
  "input": {
    "prompt": "A red panda on a mossy branch, chewing bamboo, dappled forest light...",
    "width": 768,
    "height": 512,
    "num_frames": 49,
    "frame_rate": 25.0,
    "seed": 42,
    "s3_config": {
      "region": "nyc3",
      "bucketName": "my-videos",
      "accessId": "DO00...",
      "accessSecret": "..."
    },
    "s3_prefix": "renders/2026-08"
  }
}
```

`region` is a DigitalOcean shorthand — it expands to
`https://{region}.digitaloceanspaces.com`. Pass `endpointUrl` explicitly instead for a CDN
endpoint or a non-DO S3; an explicit `endpointUrl` always wins.

For a single-bucket deployment, set `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`,
`BUCKET_SECRET_ACCESS_KEY` and `BUCKET_NAME` on the endpoint and omit `s3_config`
entirely, keeping credentials out of request payloads.

### Input parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | **required** | Single flowing paragraph, chronological, ≤200 words. |
| `width` | int | `768` | **Must be a multiple of 64.** |
| `height` | int | `512` | **Must be a multiple of 64.** |
| `num_frames` | int | `49` | **Must satisfy `(num_frames - 1) % 8 == 0`** → 49, 57, …, 121. |
| `frame_rate` | float | `25.0` | Playback fps. |
| `seed` | int | `42` | |
| `auto_duration` | bool | `false` | Predict `num_frames` from the caption. Requires the duration head. |
| `image_conditioning` | array | `[]` | `[{"url", "frame_idx", "strength", "crf"}]`. `frame_idx: 0` replaces the first frame (i2v); others act as keyframe guidance. Omit `crf` to use the model's trained value. |
| `s3_config` | object | — | See above. |
| `s3_prefix` | string | — | Key prefix inside the bucket. |

Both geometry rules are validated **before** any model is built, so a typo fails in
milliseconds rather than after several minutes of Gemma load. The error names the nearest
valid frame count.

Not accepted, by design: `negative_prompt`, `num_inference_steps`, `guidance_scale`,
`stg_scale`. The distilled pipeline runs guidance-free on a fixed 8-step (+3 refine) sigma
schedule and has no input for any of them — they belong to the full-model two-stage
pipelines.

### Response

```json
{
  "status": "success",
  "video_url": "https://my-videos.nyc3.digitaloceanspaces.com/...?X-Amz-Signature=...",
  "seed": 42,
  "num_frames": 49,
  "width": 768,
  "height": 512,
  "frame_rate": 25.0
}
```

The URL is presigned for 7 days. On failure the handler returns
`{"status": "failed", "error": ..., "traceback": ...}` and also prints the traceback to
the container log — a failure that leaves no trace in the logs is one you cannot debug
after the fact.

---

## Observability

Every job emits one JSON line to the container log:

```json
{"event": "job_timing", "job_id": "...", "model": "ltx-2.5-distilled",
 "total_s": 214.7, "stages_s": {"pipeline_init": 0.3, "input_download": 0.0,
 "generate": 188.2, "encode": 9.1, "upload": 17.1},
 "quantization": "none", "offload_mode": "cpu", "video_vae": "conv",
 "weights_staged_locally": ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
 "ltx-2.5-22b-distilled-transformer-bf16.safetensors"],
 "peak_vram_gb": 21.4, "outcome": "success", "num_frames": 49}
```

`generate` covers the work that dominates cost and is invisible in the response: rebuilding
the text encoder, both transformer builds, the upsampler and the decoders, plus the
denoising they exist to feed. `weights_staged_locally` tells you whether the reads came off
NVMe or the network volume — usually the single biggest lever on job duration.

## Why weights are staged to local disk

`ltx-pipelines` frees each component after use and rebuilds it from its safetensors on the
next call, because the pack does not fit in VRAM or RAM. The transformer is therefore
re-read **twice** per job, and 2.5 splits the old monolith into five files, so the video VAE
is now opened **four** times per job — once by `ImageConditioner` at each stage, once by
`VideoUpsampler`, once by `VideoDecoder`.

`/runpod-volume` reads at roughly 270 MB/s. Local NVMe turns those reads into seconds. The
one-time copy costs about as much as a single transformer build, is paid once per worker
lifetime, and FlashBoot amortises it across every job that worker goes on to serve.
Anything that does not fit is skipped and served from the volume, so staging is safe to
leave on regardless of container disk size.

## Repository layout

| File | Role |
|---|---|
| `handler.py` | RunPod entrypoint, request validation, pipeline cache, upload. |
| `model_paths_config.py` | Single source of truth for the split-pack layout and staging priority. |
| `download_models.py` | Populates the models volume from Hugging Face; marker-guarded. |
| `staging.py` | Copies weights onto container-local disk. Carried over unchanged from the 2.3 worker. |
| `Dockerfile` | Pinned torch / transformers / `LTX_REF`, plus a build-time API check. |

## Migrating from the 2.3 worker

Callers need no changes. Operationally:

1. New endpoint, new volume (or accept ~70 GB of 2.3 leftovers on a shared one —
   `download_models.py` reports them and deliberately does **not** delete them, since
   another endpoint may still be serving from that volume).
2. `HF_TOKEN` is now mandatory, not optional — LTX-2.5 is gated.
3. `MODELS_ROOT` layout changed from `ltx-2.3/` + `gemma-3-12b/` to `ltx-2.5/`.
4. Re-check `LTX_QUANTIZATION` / `LTX_OFFLOAD_MODE` against the GPU you deploy on.
