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
from huggingface_hub.errors import GatedRepoError

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


GATE_URL = f"https://huggingface.co/{HF_REPO}"


def verify_token(token: str | None) -> None:
    """Check the token with Hugging Face before starting a ~66 GiB gated download.

    One cheap round-trip that separates the three failures a bare 401 conflates:

      no token      -- HF_TOKEN unset or blank.
      bad token     -- set, but HF rejects the credential (401). Revoked,
                       expired, truncated on paste, or from a deleted account.
      wrong account -- credential accepted, so whoami succeeds and names the
                       account; if the download then 403s, that account simply
                       has not accepted the model terms.

    Without this, all three surface identically as a traceback out of
    hf_hub_download after the first HEAD, and the obvious-but-usually-wrong
    conclusion is "I need to accept the terms".
    """
    if not token:
        print(
            f"[models] HF_TOKEN is not set, and {HF_REPO} is gated. Set it on the "
            f"endpoint to a Read token from an account that has accepted the terms "
            f"at {GATE_URL}"
        )
        return

    try:
        from huggingface_hub import whoami

        identity = whoami(token=token)
        print(f"[models] HF token accepted -- authenticated as {identity.get('name', '<unknown>')}")
    except Exception as error:
        print(
            f"[models] HF_TOKEN was REJECTED by Hugging Face ({type(error).__name__}). "
            f"The credential itself is bad -- this is not a permissions problem. "
            f"Check for a trailing newline or truncation when it was pasted into the "
            f"endpoint env var, confirm the token has not been revoked, and that it is "
            f"a Read token (fine-grained tokens also need 'Read access to contents of "
            f"all public gated repos you can access')."
        )


def _gated_repo_help(error: Exception) -> str:
    """Turn a GatedRepoError into a message that names the actual next step."""
    return (
        f"Cannot download {HF_REPO}: the repo is gated and access was refused.\n"
        f"  401 -> the token was rejected. Re-check HF_TOKEN on the endpoint for a\n"
        f"         trailing newline, truncation, or revocation.\n"
        f"  403 -> the token is valid but its account has not accepted the model\n"
        f"         terms. Open {GATE_URL} while signed in as that account and accept.\n"
        f"Original error: {error}"
    )


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

    # .strip() is not cosmetic. Pasting a token into a dashboard env var field
    # very often carries a trailing newline or space, and Hugging Face rejects
    # the resulting Authorization header with a 401 that reads exactly like an
    # invalid token -- which sends you looking at permissions instead of
    # whitespace. Empty-after-strip collapses back to None so the not-set
    # branch still fires.
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    verify_token(token)

    files = required_repo_files()
    for index, filename in enumerate(files, start=1):
        print(f"[models] ({index}/{len(files)}) ensuring {filename}...")
        try:
            hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=LTX_DIR, token=token)
        except GatedRepoError as error:
            raise RuntimeError(_gated_repo_help(error)) from error

    _mark_complete(LTX_DIR, marker)
    print(f"[models] ltx-2.5 distilled pack complete ({VIDEO_VAE_VARIANT} VAE)")
    return LTX_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.getenv("MODELS_ROOT", "/workspace/models"))
    args = parser.parse_args()
    ensure_models(args.dir)
