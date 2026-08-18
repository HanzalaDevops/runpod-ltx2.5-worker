# Request bodies

Copy one of these, fill in the four `REPLACE_WITH_*` values, and POST it.

| File | Use |
|---|---|
| `minimal.json` | Smallest valid request. Everything else defaults. |
| `t2v.json` | Text-to-video with all common knobs set explicitly. |
| `i2v.json` | Image-to-video — first frame supplied by URL. |

## Filling in the bucket

Only four values are yours to set:

```json
"region":       "nyc3",              // or sgp1, fra1, ams3, sfo3, syd1, tor1, blr1
"bucketName":   "my-videos",
"accessId":     "DO00XXXXXXXXXXXXXXXX",
"accessSecret": "your-spaces-secret"
```

`region` expands to `https://{region}.digitaloceanspaces.com`. For a CDN endpoint or a
non-DO S3, replace `region` with `endpointUrl` instead — an explicit `endpointUrl` wins.

Prefer not to put credentials in every request? Set `BUCKET_ENDPOINT_URL`,
`BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY` and `BUCKET_NAME` on the endpoint and
drop `s3_config` from the body entirely.

## Sending it

```bash
# async (recommended -- generation takes minutes)
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @requests/t2v.json

# then poll
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://api.runpod.ai/v2/$ENDPOINT_ID/status/<job-id>"
```

`/runsync` also works but will hit the gateway timeout on anything but the shortest clips.

## The two rules you cannot break

- `width` and `height` **must be multiples of 64**.
- `num_frames` **must satisfy `(num_frames - 1) % 8 == 0`** → 49, 57, 65, 73, 81, 89, 97,
  105, 113, 121.

Both are validated before any model loads, so a mistake fails in milliseconds and the
error names the nearest valid frame count.

## Ignored if you send them

`negative_prompt`, `num_inference_steps`, `guidance_scale`, `stg_scale`. The distilled
pipeline runs guidance-free on a fixed 8-step (+3 refine) schedule and has no input for
any of them — they belong to the full-model two-stage pipelines.

## Response

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

Presigned for 7 days.
