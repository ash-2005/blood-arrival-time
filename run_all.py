"""
run_all.py
----------
Single entry point for the full blood arrival time × TVB pipeline.
Skips steps whose outputs already exist — safe to re-run.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    {
        "name":   "Step 1 — Download fMRI data",
        "script": "download_data.py",
        "check":  lambda: (Path("data/ds000228") /
                           "sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz").exists(),
    },
    {
        "name":   "Step 2 — Run rapidtide (blood arrival delay map)",
        "script": "run_rapidtide.py",
        "check":  lambda: len(list(Path("data/rapidtide_output").glob(
                               "*_desc-maxtime_map.nii.gz"))) > 0
                          if Path("data/rapidtide_output").exists() else False,
    },
    {
        "name":   "Step 3 — Parcellate delays to 100 Schaefer regions",
        "script": "parcellate_delays.py",
        "check":  lambda: Path("data/region_delays.npy").exists(),
    },
    {
        "name":   "Step 4 — Empirical FC bias quantification",
        "script": "compute_fc_bias.py",
        "check":  lambda: Path("data/fc_bias_results.npz").exists(),
    },
    {
        "name":   "Step 5 — TVB simulation with hemodynamic delay injection",
        "script": "simulate_tvb.py",
        "check":  lambda: Path("data/tvb_sim_results.npz").exists(),
    },
    {
        "name":   "Step 6 — Spatial characterisation and publication figures",
        "script": "characterise_bias.py",
        "check":  lambda: Path("figures/fig4_summary_card.png").exists(),
    },
]

BAR = "─" * 60


def fmt_size(path):
    return f"{path.stat().st_size / 1024:.0f} KB" if path.exists() else "missing"


def main():
    print(f"\n{'═'*60}")
    print("  Blood Arrival Time × TVB — Full Pipeline")
    print(f"{'═'*60}\n")

    total_start = time.time()
    step_times  = []

    for step in STEPS:
        print(f"{BAR}\n{step['name']}")
        try:
            already_done = step["check"]()
        except Exception:
            already_done = False

        if already_done:
            print("  ✓ Skipped — output already exists")
            step_times.append((step["name"], None))
            continue

        t0 = time.time()
        print(f"  Running: python {step['script']} ...")
        try:
            subprocess.run([sys.executable, "-W", "ignore", step["script"]],
                           check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n  ✗ FAILED (exit {e.returncode})")
            print("  Aborting pipeline.")
            sys.exit(1)
        elapsed = time.time() - t0
        step_times.append((step["name"], elapsed))
        print(f"  ✓ Done in {elapsed/60:.1f} min")

    total = time.time() - total_start
    print(f"\n{BAR}")
    print(f"Pipeline complete in {total/60:.1f} min total\n")

    print("Step runtimes:")
    for name, t in step_times:
        if t is None:
            print(f"  {name}: skipped")
        else:
            print(f"  {name}: {t/60:.1f} min")

    print("\nOutput files:")
    outputs = [
        Path("data/region_delays.npy"),
        Path("data/region_labels.npy"),
        Path("data/fc_bias_results.npz"),
        Path("data/parcellated_ts.npy"),
        Path("data/tvb_sim_results.npz"),
        Path("figures/fig1_delay_profile.png"),
        Path("figures/fig2_fc_bias_story.png"),
        Path("figures/fig3_network_bias_matrix.png"),
        Path("figures/fig4_summary_card.png"),
    ]
    for p in outputs:
        mark = "✓" if p.exists() else "✗"
        print(f"  {mark} {p}  ({fmt_size(p)})")

    print(f"\n{'═'*60}")


if __name__ == "__main__":
    main()
