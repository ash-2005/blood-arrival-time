"""
run_rapidtide.py
----------------
Runs rapidtide on sub-pixar001's BOLD data to produce a voxelwise blood
arrival delay (lag) map.

Uses subprocess to call the rapidtide CLI via sys.executable (guaranteed
to use the venv Python), bypassing the v3.x rapidtide_main(argparsingfunc)
API which is not directly callable with a flat argument list.

Run from the GSOC/ working directory:
    python run_rapidtide.py
"""

import subprocess
import sys
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data") / "ds000228"
OUT_DIR   = Path("data") / "rapidtide_output"
BOLD_FILE = DATA_DIR / "sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz"
MASK_FILE = DATA_DIR / "sub-pixar001_analysis_mask.nii.gz"
OUT_PREFIX = str(OUT_DIR / "sub-pixar001")


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
    # Locate rapidtide.exe inside the same venv as our Python
    scripts_dir = Path(sys.executable).parent
    exe = scripts_dir / "rapidtide.exe"
    if not exe.exists():
        exe = scripts_dir / "rapidtide"   # Linux/mac fallback
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
        "--globalmeaninclude", str(MASK_FILE),  # use brainmask for global signal
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
    # Print last 5000 chars (rapidtide output can be very long)
    suffix = out[-5000:] if len(out) > 5000 else out
    print(suffix)

    if proc.returncode != 0:
        raise RuntimeError(
            f"rapidtide exited with code {proc.returncode}. "
            "See output above for details."
        )

    print(f"\n[OK]   rapidtide finished in {elapsed:.1f} min")


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
        # list all outputs for debugging
        files = sorted(OUT_DIR.iterdir()) if OUT_DIR.exists() else []
        raise FileNotFoundError(
            "Lag map (*desc-maxtime_map.nii.gz) not found in output directory.\n"
            f"Files in {OUT_DIR}:\n" +
            "\n".join(f"  {f.name}: {f.stat().st_size/1e6:.2f} MB" for f in files)
        )

    print("\nAll rapidtide output files:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1_048_576:.2f} MB")


if __name__ == "__main__":
    main()
