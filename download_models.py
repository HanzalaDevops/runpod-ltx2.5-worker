"""Populate the LTX-2.5 split pack on the models volume.

Simpler than the 2.3 worker's equivalent in one important way: everything comes
from a single Hugging Face repo. LTX-2.5 ships the LTX-tuned Gemma 4 text
encoder *inside* Lightricks/LTX-2.5 with the text projections already packed in,
so there is no second snapshot_download of a Google Gemma repo and none of the
partial-tree repair logic that needed.

Every file is fetched individually with hf_hub_download, which is resumable and
atomic per file, so an interrupted run repairs itself on the next start.
"""

import argparse
import os

from huggingface_hub import hf_hub_download

from model_paths_config import (
    ENABLE_DURATION_HEAD,
    HF_REPO,
    LTX_DIR,
    VIDEO_VAE_VARIANT,
    required_repo_files,
)

# Versioned by file-set, not by model. A volume marked complete while serving the
# conv VAE holds no diffusion VAE, so flipping LTX_VIDEO_VAE must not be
# short-circuited by the previous marker. Bump the suffix whenever the contents
# of required_repo_files() change shape.
def completion_marker() -> str:
    suffix = "dur" if ENABLE_DURATION_HEAD else "nodur"
    return f".download-complete-ltx25-distilled-{VIDEO_VAE_VARIANT}-{suffix}-v1"


# Left behind on any volume an LTX-2.3 worker touched. The 2.3 monolith is ~46 GB
# and its Gemma-3 tree another ~24 GB; neither is loadable by a 2.5 pipeline, and
# on a shared volume they are exactly what makes the 2.5 pack fail mid-download
# with EDQUOT. Pruned by directory since the 2.3 worker wrote its own subtrees.
OBSOLETE_DIRS = ["ltx-2.3", "gemma-3-12b"]


def _is_complete(directory: str, marker_name: str) -> bool:
    return os.path.exists(os.path.join(directory, marker_name))


def _mark_complete(directory: str, marker_name: str) -> None:
    with open(os.path.join(directory, marker_name), "w") as marker:
        marker.write("ok\n")


def _prune_obsolete(models_root: str) -> None:
    """Report -- but never delete -- weights from an older deployment.

    Deliberately non-destructive. A models volume is often shared between the 2.3
    and 2.5 endpoints during a migration, and silently reclaiming 70 GB out from
    under a live 2.3 worker turns a disk-space warning into an outage. Surfacing
    the number is enough for an operator to decide.
    """
    import shutil

    for name in OBSOLETE_DIRS:
        path = os.path.join(models_root, name)
        if not os.path.isdir(path):
            continue
        size_gb = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(path)
            for f in files
            if not os.path.islink(os.path.join(root, f))
        ) / 1e9
        free_gb = shutil.disk_usage(models_root).free / 1e9
        print(
            f"[models] found LTX-2.3 leftovers at {path} ({size_gb:.1f} GB). "
            f"{free_gb:.1f} GB free. Not removing them -- another endpoint may "
            f"still be serving from this volume. Delete manually if space is tight."
        )


def ensure_models(target_dir: str) -> str:
    """Download the LTX-2.5 distilled split pack into target_dir.

    When target_dir is a mounted network volume the files persist across workers,
    so the first cold start populates it and every later start reuses it instead
    of re-pulling ~66 GiB from Hugging Face.

    Completion is tracked with a marker written only after every file returns
    successfully. Never infer completion from the presence of an individual file:
    the small ones land in milliseconds while the 44 GB transformer takes
    minutes, and a per-file check would report "already present" forever after an
    interruption in that window.

    LTX-2.5 is a gated repo. Without an HF_TOKEN whose account has accepted the
    model terms, the first download fails with 401/403 -- which is why this runs
    at worker startup and not lazily inside a job, where it would surface as a
    failed generation instead of a failed boot.
    """
    models_dir = os.path.abspath(target_dir)
    os.makedirs(LTX_DIR, exist_ok=True)

    _prune_obsolete(models_dir)

    marker = completion_marker()
    if _is_complete(LTX_DIR, marker):
        print(f"[models] ltx-2.5 distilled pack complete ({VIDEO_VAE_VARIANT} VAE), skipping")
        return LTX_DIR

    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "[models] HF_TOKEN is not set. Lightricks/LTX-2.5 is a gated repo; "
            "if the download 401s, set HF_TOKEN to a Read token from an account "
            "that has accepted the model terms."
        )

    files = required_repo_files()
    for index, filename in enumerate(files, start=1):
        print(f"[models] ({index}/{len(files)}) ensuring {filename}...")
        hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=LTX_DIR, token=token)

    _mark_complete(LTX_DIR, marker)
    print(f"[models] ltx-2.5 distilled pack complete ({VIDEO_VAE_VARIANT} VAE)")
    return LTX_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.getenv("MODELS_ROOT", "/workspace/models"))
    args = parser.parse_args()
    ensure_models(args.dir)
