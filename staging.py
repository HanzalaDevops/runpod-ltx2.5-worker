"""Copy model weights from the network volume onto local container disk.

Carried over unchanged from the LTX-2.3 worker; the reasoning applies harder
under 2.5, not less.

ltx-pipelines rebuilds each component from its safetensors on every call because
it has nowhere to cache them: the split pack totals ~66 GiB against this
endpoint's VRAM and container RAM. The distilled transformer alone is re-read
twice per job (DiffusionStage runs stage 1 and stage 2 from the same object,
building and freeing the model each time), and 2.5 splits what used to be one
monolith into five files, so the video VAE is now opened four separate times per
job -- once by ImageConditioner at each stage, once by VideoUpsampler, once by
VideoDecoder. /runpod-volume reads at roughly 270 MB/s, so those reads dominate
the job far more than the denoising they feed.

Local NVMe turns them into seconds. The one-time copy costs about as much as a
single transformer build and is paid once per worker lifetime rather than
repeatedly per job, which FlashBoot then amortises across every job that worker
goes on to serve.

Targets are flattened to basenames under local_root. Safe here because the six
files in the 2.5 pack have distinct basenames even though the repo nests them
under diffusion_models/, text_encoders/, vae/ and so on.
"""

import os
import shutil
import time

# Headroom left on the container disk after staging. /tmp holds the decoded
# mp4 and any conditioning images while a job runs, and a staged copy must
# never be the thing that fills the disk out from under a live job.
DEFAULT_RESERVE_BYTES = 5 * 1024 ** 3


def _free_bytes(path):
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def _is_memory_backed(path):
    """True when path lives on tmpfs/ramfs.

    Guards against the worst possible outcome of this module: staging 46 GB
    into a RAM-backed filesystem on a box with 50 GB of RAM would OOM-kill the
    worker at startup, and it would look like a model-loading bug rather than
    a caching one. Cheaper to check than to debug.
    """
    try:
        with open("/proc/mounts") as mounts:
            entries = [line.split() for line in mounts]
    except OSError:
        return False

    target = os.path.abspath(path)
    best_match = ""
    best_type = ""
    for entry in entries:
        if len(entry) < 3:
            continue
        mount_point, fs_type = entry[1], entry[2]
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) >= len(best_match):
                best_match, best_type = mount_point, fs_type

    return best_type in ("tmpfs", "ramfs")


def _tree_size(path):
    """Total bytes of a directory's regular files, ignoring symlinks."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            entry = os.path.join(root, name)
            if not os.path.islink(entry):
                total += os.path.getsize(entry)
    return total


def _entry_size(path):
    return _tree_size(path) if os.path.isdir(path) else os.path.getsize(path)


def _discard(path):
    """Remove a staged file or directory, whichever it turned out to be."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def _copy(source, target, size):
    """Copy one file or directory, falling back to source if anything fails.

    A partial copy is worse than no copy -- it would load as a truncated
    checkpoint and fail deep inside the pipeline -- so the copy lands under a
    temp name and is only promoted once it returns successfully. Gemma arrives
    as a directory from snapshot_download while the LTX checkpoints are single
    files, so both shapes go through here.
    """
    temp = f"{target}.partial"
    _discard(temp)
    started = time.perf_counter()
    try:
        if os.path.isdir(source):
            shutil.copytree(source, temp, symlinks=True)
        else:
            shutil.copyfile(source, temp)
        # os.replace refuses to overwrite a non-empty directory, so clear any
        # previous staged copy first. Losing it is safe: the source is
        # untouched, and a failure here just falls back to the network volume.
        _discard(target)
        os.replace(temp, target)
    except OSError as error:
        print(f"[staging] failed to stage {os.path.basename(source)}: {error}")
        _discard(temp)
        return source

    elapsed = max(time.perf_counter() - started, 1e-6)
    print(
        f"[staging] staged {os.path.basename(source)} "
        f"({size / 1e9:.1f} GB in {elapsed:.0f}s, {size / 1e6 / elapsed:.0f} MB/s)"
    )
    return target


def stage_files(paths, local_root, reserve_bytes=DEFAULT_RESERVE_BYTES):
    """Copy weights onto local disk, in the caller's priority order.

    Accepts files and directories alike -- the LTX checkpoints are single
    files, Gemma is a snapshot_download tree.

    Returns {original_path: path_to_use}. Anything that does not fit maps to
    itself, so the caller transparently falls back to the network volume and
    the endpoint keeps working on a container disk too small to hold the
    checkpoint. Pass the highest-value entry first: benefit is (bytes read per
    job), not size on disk, and the distilled checkpoint is re-read twice per
    job while Gemma and the upsampler are read once. An entry that does not
    fit is skipped rather than aborting the run, so a smaller one behind it
    can still be staged.
    """
    mapping = {path: path for path in paths}

    if not paths:
        return mapping

    if _is_memory_backed(local_root):
        print(f"[staging] {local_root} is memory-backed, staging disabled")
        return mapping

    try:
        os.makedirs(local_root, exist_ok=True)
    except OSError as error:
        print(f"[staging] cannot create {local_root}: {error}, staging disabled")
        return mapping

    for source in paths:
        name = os.path.basename(source)

        if not os.path.exists(source):
            print(f"[staging] {name} not found at {source}, leaving as-is")
            continue

        size = _entry_size(source)
        target = os.path.join(local_root, name)

        # A worker resumed by FlashBoot, or restarted onto a warm container
        # disk, already has it. Size-match rather than trusting presence: a
        # copy interrupted before os.replace leaves only a .partial, but a
        # truncated real file would otherwise be reused forever. For a
        # directory this also catches a half-copied tree.
        if os.path.exists(target) and _entry_size(target) == size:
            print(f"[staging] {name} already staged")
            mapping[source] = target
            continue

        free = _free_bytes(local_root)
        if size + reserve_bytes > free:
            print(
                f"[staging] skipping {name}: needs {size / 1e9:.1f} GB plus "
                f"{reserve_bytes / 1e9:.1f} GB reserve, only {free / 1e9:.1f} GB free "
                f"-- serving it from the network volume instead"
            )
            continue

        mapping[source] = _copy(source, target, size)

    return mapping
