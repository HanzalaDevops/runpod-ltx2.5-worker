"""Resumable HTTP download with a deterministic partial path.

Why this exists instead of hf_hub_download.

huggingface_hub writes its scratch file to a path carrying a per-attempt random
suffix, so a killed download is never resumed -- the next attempt starts from
byte zero and orphans the previous partial. On a Pod that is merely wasteful. On
a serverless worker, where a 44 GB file does not fit inside the startup budget,
it is fatal: the worker is killed part-way every single time, so the file can
never complete no matter how many times it retries. Observed in production: 16
partials, 232 GB, one 44 GB file, zero completions.

Seeding the volume from a Pod sidesteps it, but that assumes a Pod is available
in the volume's region -- which is not always true, and a network volume cannot
move regions.

So: one deterministic ``<file>.partial`` per target, an HTTP Range request from
whatever byte it already holds, and an atomic rename on completion. Every worker
lifetime now *adds* progress instead of discarding it, and a file that needs
more time than one startup budget allows completes across several restarts.
"""

import os
import time

import requests

CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_EVERY_BYTES = 2 * 1024 ** 3
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 120


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def remote_size(url, token):
    """Total size of the target, or None when the server will not say.

    Follows redirects so the Content-Length comes from the CDN object rather
    than the 302 from the API host.
    """
    response = requests.head(
        url, headers=_auth_headers(token), allow_redirects=True,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    )
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    return int(length) if length else None


def download_resumable(url, destination, token=None, label=None):
    """Fetch ``url`` to ``destination``, continuing a prior partial if present.

    Returns the number of bytes transferred by *this* call, so a caller can tell
    "already complete" (0) from "made progress" without stat-ing anything.

    The partial path is derived from the destination and nothing else. That is
    the whole point: it must be identical across restarts for the Range request
    to have anything to resume from.
    """
    label = label or os.path.basename(destination)
    total = remote_size(url, token)

    if os.path.exists(destination):
        have = os.path.getsize(destination)
        if total is None or have == total:
            print(f"[dl] {label}: already complete ({have / 1e9:.1f} GB)")
            return 0
        print(f"[dl] {label}: size mismatch ({have} != {total}), re-fetching")
        os.remove(destination)

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    partial = destination + ".partial"
    resume_from = os.path.getsize(partial) if os.path.exists(partial) else 0

    if total is not None and resume_from > total:
        print(f"[dl] {label}: partial is larger than the target, discarding")
        os.remove(partial)
        resume_from = 0

    if total is not None and resume_from == total:
        os.replace(partial, destination)
        print(f"[dl] {label}: partial was already complete, promoted")
        return 0

    headers = _auth_headers(token)
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        pct = f" ({100 * resume_from / total:.0f}%)" if total else ""
        print(f"[dl] {label}: resuming at {resume_from / 1e9:.1f} GB{pct}")
    else:
        size_note = f" of {total / 1e9:.1f} GB" if total else ""
        print(f"[dl] {label}: starting{size_note}")

    with requests.get(
        url, headers=headers, stream=True, allow_redirects=True,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    ) as response:
        response.raise_for_status()

        # 200 to a Range request means the server ignored it and is sending the
        # whole object. Appending would corrupt the file, so truncate instead.
        if resume_from and response.status_code != 206:
            print(f"[dl] {label}: server ignored Range (HTTP {response.status_code}), restarting")
            resume_from = 0

        mode = "ab" if resume_from else "wb"
        written = resume_from
        since_report = 0
        started = time.time()

        with open(partial, mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                since_report += len(chunk)
                if since_report >= PROGRESS_EVERY_BYTES:
                    since_report = 0
                    rate = (written - resume_from) / max(time.time() - started, 1e-6) / 1e6
                    pct = f" ({100 * written / total:.0f}%)" if total else ""
                    print(f"[dl] {label}: {written / 1e9:.1f} GB{pct} at {rate:.0f} MB/s", flush=True)

    if total is not None and written != total:
        # Do NOT delete the partial -- it is exactly what the next attempt resumes
        # from, and that is the entire mechanism keeping this survivable.
        raise IOError(
            f"{label}: transfer ended at {written / 1e9:.1f} GB of {total / 1e9:.1f} GB. "
            f"Partial kept for resume."
        )

    os.replace(partial, destination)
    print(f"[dl] {label}: complete ({written / 1e9:.1f} GB)")
    return written - resume_from
