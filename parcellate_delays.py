"""
parcellate_delays.py
--------------------
1. Loads the rapidtide voxelwise lag map
2. Fetches the Schaefer 2018 100-parcel atlas via nilearn
3. Resamples lag map to atlas space
4. Averages delay per parcel (NaN regions → median imputation)
5. Saves data/region_delays.npy and data/region_labels.npy
6. Generates figures/phase1_delay_distribution.png
7. Writes report.md

Run from the GSOC/ working directory:
    python parcellate_delays.py
"""

from pathlib import Path
import textwrap

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nilearn import datasets, image

OUT_DIR       = Path("data") / "rapidtide_output"
DATA_DIR      = Path("data")
FIGURES_DIR   = Path("figures")
DELAYS_PATH   = DATA_DIR / "region_delays.npy"
LABELS_PATH   = DATA_DIR / "region_labels.npy"
FIGURE_PATH   = FIGURES_DIR / "phase1_delay_distribution.png"
REPORT_PATH   = Path("report.md")

BOLD_FILE     = (Path("data") / "ds000228" /
                 "sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz")


def find_lag_map() -> Path:
    candidates = list(OUT_DIR.glob("*desc-maxtime_map.nii.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No lag map (*_desc-maxtime_map.nii.gz) found in {OUT_DIR}. "
            "Run run_rapidtide.py first."
        )
    return candidates[0]


def load_lag_map(lag_path: Path):
    print(f"Loading lag map: {lag_path}")
    img = nib.load(str(lag_path))
    data = img.get_fdata(dtype=np.float32)
    print(f"  Shape: {data.shape}, voxels with data: {np.isfinite(data).sum()}")
    return img, data


def fetch_atlas():
    print("Fetching Schaefer 2018 atlas (100 parcels, 7 networks) ...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(atlas["maps"])
    all_labels = [lbl.decode() if isinstance(lbl, bytes) else lbl
                  for lbl in atlas["labels"]]
    labels = all_labels[1:]
    print(f"  Atlas shape: {atlas_img.shape}, brain parcels: {len(labels)}")
    return atlas_img, labels


def resample_lag_to_atlas(lag_img, atlas_img):
    print("Resampling lag map to atlas space ...")
    lag_resampled = image.resample_to_img(
        lag_img, atlas_img,
        interpolation="linear",
        copy=True,
    )
    print(f"  Resampled lag shape: {lag_resampled.shape}")
    return lag_resampled


def parcellate(lag_resampled, atlas_img, labels):
    print("Parcellating ...")
    lag_data    = lag_resampled.get_fdata(dtype=np.float32)
    atlas_data  = np.asarray(atlas_img.dataobj, dtype=np.int32)

    n_regions = len(labels)
    delays = np.full(n_regions, np.nan)

    for i, label in enumerate(labels):
        parcel_id = i + 1  # Schaefer parcels are 1-indexed
        mask = atlas_data == parcel_id
        voxels = lag_data[mask]
        valid  = voxels[np.isfinite(voxels) & (voxels != 0)]
        if len(valid) > 0:
            delays[i] = float(np.mean(valid))
        else:
            delays[i] = np.nan

    n_nan = int(np.isnan(delays).sum())
    if n_nan > 0:
        median_delay = float(np.nanmedian(delays))
        delays = np.where(np.isnan(delays), median_delay, delays)
        print(f"  Imputed {n_nan} NaN regions with median ({median_delay:.3f} s)")
    else:
        print("  No NaN regions — all parcels had valid voxels")
        median_delay = float(np.nanmedian(delays))
        n_nan = 0

    return delays, n_nan, median_delay if n_nan > 0 else None


def print_stats(delays, labels):
    print(f"  Min:    {delays.min():.4f} s  ({labels[np.argmin(delays)]})")
    print(f"  Max:    {delays.max():.4f} s  ({labels[np.argmax(delays)]})")
    print(f"  Mean:   {delays.mean():.4f} s")
    print(f"  Std:    {delays.std():.4f} s")
    print(f"  Median: {np.median(delays):.4f} s")

    top5_idx = np.argsort(delays)[-5:][::-1]
    bot5_idx = np.argsort(delays)[:5]

    print("\n  Top-5 highest-delay regions:")
    for i in top5_idx:
        print(f"    {delays[i]:+.4f} s  {labels[i]}")

    print("\n  Top-5 lowest-delay regions:")
    for i in bot5_idx:
        print(f"    {delays[i]:+.4f} s  {labels[i]}")

    return top5_idx, bot5_idx


def make_figure(delays, labels):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figure → {FIGURE_PATH}")

    sort_idx = np.argsort(delays)
    sorted_delays = delays[sort_idx]
    sorted_labels = [labels[i] for i in sort_idx]

    norm = mcolors.Normalize(vmin=sorted_delays.min(), vmax=sorted_delays.max())
    cmap = plt.cm.RdYlGn_r
    colors = cmap(norm(sorted_delays))

    fig, (ax_bar, ax_hist) = plt.subplots(
        1, 2,
        figsize=(18, 14),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax_bar, ax_hist):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    y_pos = np.arange(len(sorted_labels))
    bars = ax_bar.barh(y_pos, sorted_delays, color=colors, edgecolor="none", height=0.8)
    ax_bar.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(sorted_labels, fontsize=5.5, color="#c9d1d9")
    ax_bar.set_xlabel("Blood arrival delay (s)", color="#c9d1d9", fontsize=11)
    ax_bar.set_title("Per-region blood arrival delay\n(Schaefer 100, sorted)",
                     color="#e6edf3", fontsize=13, pad=10)
    ax_bar.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bar, shrink=0.4, pad=0.01)
    cbar.set_label("Delay (s)", color="#c9d1d9", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

    mean_d   = delays.mean()
    median_d = np.median(delays)
    ax_hist.hist(delays, bins=20, color="#388bfd", edgecolor="#161b22", alpha=0.85)
    ax_hist.axvline(mean_d,   color="#f85149", linewidth=1.5, linestyle="-",  label=f"Mean {mean_d:.3f} s")
    ax_hist.axvline(median_d, color="#ffa657", linewidth=1.5, linestyle="--", label=f"Median {median_d:.3f} s")
    ax_hist.set_xlabel("Blood arrival delay (s)", color="#c9d1d9", fontsize=11)
    ax_hist.set_ylabel("Number of regions",       color="#c9d1d9", fontsize=11)
    ax_hist.set_title("Delay distribution",       color="#e6edf3", fontsize=13, pad=10)
    legend = ax_hist.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d")
    for text in legend.get_texts():
        text.set_color("#c9d1d9")

    fig.suptitle(
        "Phase 1 — Blood arrival delay map (OpenNeuro ds000228, sub-pixar001)\n"
        "Rapidtide lag map parcellated into Schaefer 2018 100-region atlas",
        color="#e6edf3", fontsize=12, y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {FIGURE_PATH} ({FIGURE_PATH.stat().st_size / 1024:.0f} KB)")


def write_report(delays, labels, top5_idx, bot5_idx, lag_path, n_nan, median_imputed):
    print(f"\nWriting {REPORT_PATH} ...")

    bold_size = BOLD_FILE.stat().st_size / 1_048_576 if BOLD_FILE.exists() else 0
    lag_size  = lag_path.stat().st_size / 1_048_576

    top5_lines = "\n".join(
        f"  - {labels[i]}: {delays[i]:+.4f} s" for i in top5_idx
    )
    bot5_lines = "\n".join(
        f"  - {labels[i]}: {delays[i]:+.4f} s" for i in bot5_idx
    )

    nan_note = (
        f"{n_nan} regions had no valid voxels in the lag map and were imputed "
        f"with the parcel-wise median ({median_imputed:.4f} s)."
        if n_nan else
        "All 100 parcels had valid voxels in the lag map. No imputation needed."
    )

    report = textwrap.dedent(f"""\
    # Phase 1 Report — Environment + Data + Rapidtide Delay Map

    ## What it built

    Three Python scripts in the GSOC/ working directory:

    - **download_data.py** — downloads preprocessed fMRI data for sub-pixar001
      from OpenNeuro ds000228 via HTTPS (no datalad, no git-annex).
    - **run_rapidtide.py** — runs rapidtide to produce a voxelwise blood arrival
      delay (lag) map from the BOLD signal.
    - **parcellate_delays.py** — parcellates the lag map into 100 brain regions
      using the Schaefer 2018 atlas and saves the delay vector.

    ## What it did

    ### Downloads (download_data.py)
    Files saved to `data/ds000228/`:

    | File | Size (MB) |
    |------|-----------|
    | sub-pixar001_task-pixar_run-001_swrf_bold.nii.gz | {bold_size:.1f} |
    | sub-pixar001_analysis_mask.nii.gz | — |
    | sub-pixar001_task-pixar_run-001_nuisance_regressors.mat | — |

    ### Rapidtide (run_rapidtide.py)
    - Version: 3.1.8
    - Parameters: `--filterband lfo --searchrange -5 5 --passes 3 --nprocs 8 --brainmask <mask>`
    - Output lag map: `{lag_path.name}` ({lag_size:.1f} MB)

    ### Parcellation (parcellate_delays.py)
    - Atlas: Schaefer 2018, 100 parcels, 7 Yeo networks, 2 mm resolution
    - {nan_note}

    **Delay statistics (100 regions):**
    | Stat | Value (s) |
    |------|-----------|
    | Min | {delays.min():.4f} |
    | Max | {delays.max():.4f} |
    | Mean | {delays.mean():.4f} |
    | Std | {delays.std():.4f} |
    | Median | {np.median(delays):.4f} |

    **Top-5 highest-delay regions:**
    {top5_lines}

    **Top-5 lowest-delay regions:**
    {bot5_lines}

    ## Problems faced

    *(To be filled in after run — document any errors, warnings, or unexpected behaviour
    encountered during download, rapidtide execution, or parcellation.)*

    ## How problems were solved

    *(To be filled in after run — document exact fixes, fallbacks, and justifications.)*

    ## Results

    - `data/region_delays.npy` — shape (100,), blood arrival delay per region (seconds)
    - `data/region_labels.npy` — shape (100,), Schaefer region name strings
    - `figures/phase1_delay_distribution.png` — 2-panel figure:
        - Left: horizontal bar chart of all 100 regions sorted by delay,
          coloured green (low) → red (high), x-axis "Blood arrival delay (s)"
        - Right: histogram of the delay distribution with mean (red) and median (orange) lines
    """)

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  {REPORT_PATH} written ({REPORT_PATH.stat().st_size} bytes)")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    lag_path      = find_lag_map()
    lag_img, _    = load_lag_map(lag_path)
    atlas_img, labels = fetch_atlas()
    lag_resampled = resample_lag_to_atlas(lag_img, atlas_img)

    labels_arr    = np.array(labels)
    delays, n_nan, median_imputed = parcellate(lag_resampled, atlas_img, labels)

    np.save(DELAYS_PATH, delays)
    np.save(LABELS_PATH, labels_arr)
    print(f"\nSaved: {DELAYS_PATH}  shape={delays.shape}")
    print(f"Saved: {LABELS_PATH}  shape={labels_arr.shape}")

    top5_idx, bot5_idx = print_stats(delays, labels)
    make_figure(delays, labels)
    write_report(delays, labels, top5_idx, bot5_idx, lag_path, n_nan, median_imputed)

    print("\n✓ Phase 1 complete.")


if __name__ == "__main__":
    main()
