"""
run_rapidtide.py
----------------
Runs rapidtide on two BOLD inputs:

1. Preprocessed BOLD (SPM swrf, MNI space) — original Phase 1 run.
2. Raw BOLD (native scanner space) — added per mentor feedback that
   rapidtide should be applied before nuisance regression.

The raw run produces `data/rapidtide_raw/rapidtide_raw_out_desc-maxtime_map.nii.gz`
and the parcellated delays are saved to `data/region_delays_raw.npy`.

Note: the raw BOLD is in native scanner space, not MNI. The Schaefer atlas
is in MNI space, so direct parcellation is approximate (no registration).
This is documented as a limitation — a proper pipeline would register the
raw rapidtide lag map to MNI before parcellating.

Run from the GSOC/ working directory:
    python run_rapidtide.py
"""

import subprocess
import sys
import time
from pathlib import Path

DATA_DIR  = Path("data") / "ds000228"
OUT_DIR   = Path("data") / "rapidtide_output"
BOLD_FILE = DATA_DIR / "sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz"
MASK_FILE = DATA_DIR / "sub-pixar001_analysis_mask.nii.gz"
OUT_PREFIX = str(OUT_DIR / "sub-pixar001")

RAW_BOLD_FILE = DATA_DIR / "sub-pixar001_task-pixar_run-001_bold.nii.gz"
RAW_OUT_DIR   = Path("data") / "rapidtide_raw"
RAW_OUT_PREFIX = str(RAW_OUT_DIR / "rapidtide_raw_out")


def check_inputs() -> None:
    for p in (BOLD_FILE, MASK_FILE):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run download_data.py first."
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_lag_map() -> Path | None:
    candidates = list(OUT_DIR.glob("*desc-maxtime_map.nii.gz"))
    return candidates[0] if candidates else None


def already_done() -> bool:
    lag = find_lag_map()
    if lag:
        print(f"[SKIP] Lag map already exists: {lag}")
        return True
    return False


def run_rapidtide() -> None:
    """
    Call rapidtide.exe directly from the venv Scripts directory.
    """
    scripts_dir = Path(sys.executable).parent
    exe = scripts_dir / "rapidtide.exe"
    if not exe.exists():
        exe = scripts_dir / "rapidtide"
    if not exe.exists():
        raise FileNotFoundError(f"rapidtide executable not found in {scripts_dir}")

    cmd = [
        str(exe),
        str(BOLD_FILE),
        OUT_PREFIX,
        "--datatstep",    "2.0",        # TR not stored in SPM NIfTI header
        "--filterband",  "lfo",
        "--searchrange", "-5", "5",
        "--passes",      "3",
        "--nprocs",      "1",           # Windows multiprocessing spawn fails with >1
        "--brainmask",   str(MASK_FILE),
        "--globalmeaninclude", str(MASK_FILE),
        "--noprogressbar",
    ]

    print(f"[RUN]  {' '.join(str(c) for c in cmd)}\n")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = (time.time() - t0) / 60

    out = proc.stdout or ""
    suffix = out[-5000:] if len(out) > 5000 else out
    print(suffix)

    if proc.returncode != 0:
        raise RuntimeError(
            f"rapidtide exited with code {proc.returncode}. "
            "See output above for details."
        )

    print(f"\n[OK]   rapidtide finished in {elapsed:.1f} min")


def run_rapidtide_raw() -> None:
    """
    Second rapidtide pass on the raw (non-preprocessed) BOLD.
    No brain mask — raw BOLD is in native scanner space.
    """
    if not RAW_BOLD_FILE.exists():
        print("\n[SKIP] Raw BOLD not found — run download_data.py first.")
        return

    raw_lag_candidates = list(RAW_OUT_DIR.glob("*desc-maxtime_map.nii.gz"))
    if raw_lag_candidates:
        print(f"[SKIP] Raw lag map already exists: {raw_lag_candidates[0]}")
        return

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(sys.executable).parent
    exe = scripts_dir / "rapidtide.exe"
    if not exe.exists():
        exe = scripts_dir / "rapidtide"
    if not exe.exists():
        raise FileNotFoundError(f"rapidtide executable not found in {scripts_dir}")

    cmd = [
        str(exe),
        str(RAW_BOLD_FILE),
        RAW_OUT_PREFIX,
        "--datatstep",    "2.0",  # TR not stored in SPM NIfTI header
        "--filterband",  "lfo",
        "--searchrange", "-5", "5",
        "--passes",      "3",
        "--nprocs",      "1",     # Windows multiprocessing spawn fails with >1
        "--noprogressbar",
    ]

    print(f"\n[RUN raw]  {' '.join(str(c) for c in cmd)}\n")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = (time.time() - t0) / 60

    out = proc.stdout or ""
    suffix = out[-5000:] if len(out) > 5000 else out
    print(suffix)

    if proc.returncode != 0:
        print(f"  WARNING: raw rapidtide exited with code {proc.returncode}.")
        print("  This may be due to raw BOLD being in native space without a mask.")
        print("  Delay comparison will use preprocessed delays only.")
        return

    print(f"\n[OK]   raw rapidtide finished in {elapsed:.1f} min")


def parcellate_raw_delays() -> None:
    """
    Inline parcellation of raw rapidtide lag map → data/region_delays_raw.npy.
    Uses same Schaefer 100 atlas. The raw BOLD is in native space, so
    this parcellation is approximate without MNI registration.
    """
    import numpy as np
    import nibabel as nib
    from nilearn import datasets, image

    raw_lag_candidates = list(RAW_OUT_DIR.glob("*desc-maxtime_map.nii.gz"))
    if not raw_lag_candidates:
        print("  [SKIP] No raw lag map found — skipping parcellation.")
        return

    out_path = Path("data/region_delays_raw.npy")
    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists.")
    else:
        lag_path = raw_lag_candidates[0]
        print(f"\nParcellating raw lag map: {lag_path}")
        lag_img  = nib.load(str(lag_path))
        lag_data = lag_img.get_fdata(dtype=np.float32)

        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
        atlas_img = nib.load(atlas["maps"])
        lag_res   = image.resample_to_img(lag_img, atlas_img, interpolation="linear", copy=True)
        lag_res_data = lag_res.get_fdata(dtype=np.float32)
        atlas_data   = np.asarray(atlas_img.dataobj, dtype=np.int32)
        all_labels   = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in atlas["labels"]]
        labels       = all_labels[1:]

        delays_raw = np.full(100, np.nan)
        for i in range(100):
            mask  = atlas_data == (i + 1)
            voxels = lag_res_data[mask]
            valid  = voxels[np.isfinite(voxels) & (voxels != 0)]
            delays_raw[i] = float(np.mean(valid)) if len(valid) > 0 else np.nan

        n_nan = int(np.isnan(delays_raw).sum())
        if n_nan > 0:
            med = float(np.nanmedian(delays_raw))
            delays_raw = np.where(np.isnan(delays_raw), med, delays_raw)
            print(f"  Imputed {n_nan} NaN parcels with median {med:.3f} s")

        np.save(str(out_path), delays_raw)
        print(f"  Saved: {out_path}")

    # Compare raw vs preprocessed delays
    preprocessed_path = Path("data/region_delays.npy")
    raw_delays = np.load(str(out_path))
    if preprocessed_path.exists():
        prep_delays = np.load(str(preprocessed_path))
        print("\n── Delay comparison: raw vs preprocessed rapidtide ──")
        print(f"  {'Metric':<12}  {'Raw':>10}  {'Preprocessed':>14}")
        print(f"  {'Mean':<12}  {raw_delays.mean():>10.4f}  {prep_delays.mean():>14.4f} s")
        print(f"  {'Std':<12}  {raw_delays.std():>10.4f}  {prep_delays.std():>14.4f} s")
        print(f"  {'Min':<12}  {raw_delays.min():>10.4f}  {prep_delays.min():>14.4f} s")
        print(f"  {'Max':<12}  {raw_delays.max():>10.4f}  {prep_delays.max():>14.4f} s")


def main() -> None:
    check_inputs()

    if already_done():
        pass
    else:
        run_rapidtide()

    lag = find_lag_map()
    if lag:
        size_mb = lag.stat().st_size / 1_048_576
        print(f"\nLag map : {lag}")
        print(f"Size    : {size_mb:.1f} MB")
    else:
        files = sorted(OUT_DIR.iterdir()) if OUT_DIR.exists() else []
        raise FileNotFoundError(
            "Lag map (*desc-maxtime_map.nii.gz) not found in output directory.\n"
            f"Files in {OUT_DIR}:\n" +
            "\n".join(f"  {f.name}: {f.stat().st_size/1e6:.2f} MB" for f in files)
        )

    print("\nAll rapidtide output files:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1_048_576:.2f} MB")

    print("\n" + "─" * 60)
    print("Running rapidtide on raw (non-preprocessed) BOLD ...")
    run_rapidtide_raw()
    parcellate_raw_delays()


if __name__ == "__main__":
    main()
