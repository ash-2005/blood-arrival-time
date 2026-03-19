"""
download_data.py
----------------
Downloads the fMRI data for sub-pixar001 from OpenNeuro ds000228.

Downloads two sets of files:
1. Preprocessed SPM derivatives (MNI-space, nuisance-regressed BOLD)
2. Raw BOLD and T1w from the BIDS root (native space, motion-corrected only)

The raw BOLD is needed to run rapidtide on minimally-processed data, as
recommended: rapidtide estimates are distorted by global signal regression.

Run from the GSOC/ working directory:
    python download_data.py
"""

import time
import requests
from pathlib import Path

BASE_DERIV = "https://s3.amazonaws.com/openneuro.org/ds000228/derivatives/fmriprep/sub-pixar001/"
BASE_RAW   = "https://s3.amazonaws.com/openneuro.org/ds000228/sub-pixar001/"

# Files confirmed to exist in the S3 bucket
FILES_DERIV = [
    (
        BASE_DERIV,
        "sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz",
        "sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz",
    ),
    (
        BASE_DERIV,
        "sub-pixar001_analysis_mask.nii.gz",
        "sub-pixar001_analysis_mask.nii.gz",
    ),
    (
        BASE_DERIV,
        "sub-pixar001_task-pixar_run-001_ART_and_CompCor_nuisance_regressors.mat",
        "sub-pixar001_task-pixar_run-001_nuisance_regressors.mat",
    ),
]

FILES_RAW = [
    (
        BASE_RAW,
        "func/sub-pixar001_task-pixar_run-001_bold.nii.gz",
        "sub-pixar001_task-pixar_run-001_bold.nii.gz",
    ),
    (
        BASE_RAW,
        "anat/sub-pixar001_T1w.nii.gz",
        "sub-pixar001_T1w.nii.gz",
    ),
]

OUT_DIR    = Path("data") / "ds000228"
MAX_RETRY  = 10
CHUNK_SIZE = 1 << 20  # 1 MB


def get_remote_size(url: str) -> int:
    """Return Content-Length from HEAD, or 0 if unavailable."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def download_file(url: str, dest: Path) -> None:
    """
    Stream-download url to dest with:
      - Skip if already complete
      - HTTP Range resume if partially downloaded
      - Automatic retry on connection reset (up to MAX_RETRY times)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_size = get_remote_size(url)

    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"  [SKIP] {dest.name} ({dest.stat().st_size / 1_048_576:.1f} MB already downloaded)")
        return

    for attempt in range(1, MAX_RETRY + 1):
        existing = dest.stat().st_size if dest.exists() else 0

        if existing and expected_size and existing >= expected_size:
            print(f"  [DONE] {dest.name} already complete at {existing / 1_048_576:.1f} MB")
            return

        headers = {}
        mode = "ab"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            print(f"  [RESUME attempt {attempt}] {dest.name} from {existing / 1_048_576:.1f} MB ...", flush=True)
        else:
            mode = "wb"
            print(f"  [DOWN attempt {attempt}] {dest.name} ...", flush=True)

        try:
            resp = requests.get(url, stream=True, timeout=120, headers=headers)
            if resp.status_code not in (200, 206):
                resp.raise_for_status()

            downloaded = existing
            with dest.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded / expected_size * 100) if expected_size else 0
                        print(f"\r    {downloaded / 1_048_576:.1f} MB  ({pct:.0f}%)", end="", flush=True)

            print()
            final_size = dest.stat().st_size
            print(f"  [OK]   {dest.name} — {final_size / 1_048_576:.1f} MB")
            return

        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                Exception) as exc:
            print(f"\n  [WARN] attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRY:
                wait = 5 * attempt
                print(f"         retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to download {dest.name} after {MAX_RETRY} attempts") from exc


def main() -> None:
    all_groups = [
        ("Preprocessed derivatives", FILES_DERIV),
        ("Raw BOLD + anatomical",    FILES_RAW),
    ]
    total = sum(len(files) for _, files in all_groups)
    print(f"Downloading {total} files to {OUT_DIR}/\n")

    for group_name, files in all_groups:
        print(f"\n── {group_name} ──")
        for base_url, s3_name, local_name in files:
            url  = base_url + s3_name
            dest = OUT_DIR / local_name
            download_file(url, dest)

    print("\nAll files downloaded.")
    print("\nSummary:")
    for _, files in all_groups:
        for _base, _s3, local_name in files:
            p = OUT_DIR / local_name
            if p.exists():
                print(f"  {p.name}: {p.stat().st_size / 1_048_576:.1f} MB")
            else:
                print(f"  {p.name}: MISSING")


if __name__ == "__main__":
    main()
