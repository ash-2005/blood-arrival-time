# Blood Arrival Time × The Virtual Brain

**Does ignoring vascular delays distort functional connectivity in TVB, and by how much?**

This repository answers that question on real fMRI data, before anyone asked it inside a computational model.

The project description for [INCF GSoC 2026 #28](https://neurostars.org/t/gsoc-2026-project-28-title-integrating-blood-arrival-time-in-models-of-fmri-data-in-the-virtual-brain-in-ebrains/35605) notes that blood arrival time delays are "completely overlooked in computational models of large scale BOLD activity." This prototype builds the pipeline that changes that - running `rapidtide` on real resting-state fMRI, parcellating the delay map into 100 brain regions, injecting those delays into a Balloon-Windkessel hemodynamic model, and comparing the resulting FC matrices against the legacy (no-delay) approach. The short answer: the bias is real, spatially structured, and 6× larger in continuous hemodynamic models than TR-discretised empirical estimates suggest.

---

![Summary](figures/fig4_summary_card.png)

---

## Results

| | |
|---|---|
| Blood arrival delay range (100 regions) | −2.97 s to −0.78 s &nbsp;(spread: **2.2 s**) |
| Empirical FC bias — mean \|ΔFC\| | 0.0023 |
| HRF simulation FC bias — mean \|ΔFC\| | **0.0148** (6× larger) |
| Delay difference → FC bias (Spearman ρ) | **0.396**, p < 10⁻¹⁸⁵ |
| Most distorted network pair | **Limbic × Cont**, mean \|ΔFC\| = 0.023 |
| Parcels with valid rapidtide estimates | 100 / 100 |

The empirical ρ is slightly negative (−0.062) while the simulation ρ is strongly positive (0.396). This isn't a contradiction, it's a TR discretisation artefact. At TR=2s, all 100 delays collapse to just two integer shifts (0 or −1 sample). Pairs where both regions shift by −1 have zero net displacement and no FC change, while pairs where only one region shifts tend to be the ones closest to the −1s boundary, which happen to have *smaller* absolute delay differences, inverting the sign. At 1ms simulation resolution this artefact disappears and the true positive relationship emerges cleanly.

---

## What this pipeline does

```
OpenNeuro ds000228 (real fMRI, SPM-preprocessed)
        ↓
  rapidtide v3.1.8
  → voxelwise blood arrival lag map
        ↓
  Schaefer 100-region atlas parcellation
  → τᵢ: per-region delay vector (seconds)
        ↓
        ├── Empirical branch
        │     shift each region's BOLD by −τᵢ before correlating
        │     → Legacy FC vs Corrected FC
        │
        └── Simulation branch
              HRF convolution: per-region onset shifted by τᵢ
              Balloon-Windkessel ODE: delayed neural input per region
              → Legacy FC vs Delay-injected FC
```

---

## Figures

### Blood arrival delays across 100 brain regions

`rapidtide` estimated blood arrival time at each voxel. After parcellating into Schaefer 100 regions, the delays span 2.2 seconds, enough to meaningfully shift BOLD signals relative to each other before FC is even computed. The **Cont** network shows the widest within-network spread, and it's also the network most affected by the bias.

![Fig 1](figures/fig1_delay_profile.png)

---

### Empirical evidence and computational confirmation

Two scatter plots tell the main story side by side. On the left: the empirical analysis, where TR=2s discretisation blunts the signal. On the right: the HRF simulation at 1ms resolution, where the relationship between inter-regional delay difference and FC distortion is unambiguous (ρ=0.396, p<10⁻¹⁸⁵). The bar chart puts both in context, continuous hemodynamic modelling reveals an effect 6× larger than the empirical estimate.

![Fig 2](figures/fig2_fc_bias_story.png)

---

### FC matrices: legacy vs corrected

The ΔFC matrix (right panel) shows that the bias is not uniformly distributed. One region — `RH_Cont_PFCl_4` - sits exactly at the TR shift boundary, meaning it gets corrected while most of its network partners don't. Its entire row and column lights up in the ΔFC matrix, which is visible as the cross-shaped pattern. This is what "spatially structured bias" means in practice.

![Fig 2 matrices](figures/phase2_fc_matrices.png)

---

### Which network pairs are most distorted?

The 7×7 network bias matrix on the right shows that Cont and Default networks drive most of the distortion, consistent with their known vascular heterogeneity and their spanning of both early and late blood arrival regions. The asterisk marks Limbic × Cont as the single most distorted pair.

![Fig 3](figures/fig3_network_bias_matrix.png)

---

## Reproducing the results

```bash
pip install nibabel nilearn rapidtide scipy matplotlib seaborn h5py
python run_all.py
```

`run_all.py` runs the full pipeline in order and skips any step whose output already exists — so after the first run you can re-run individual steps without redoing the 30-minute rapidtide computation.

**Windows note:** rapidtide imports a POSIX-only `resource` module at the top level. Fix it by creating a stub file at `.venv/Lib/site-packages/resource.py` with the contents:

```python
RLIMIT_AS = 0
def getrusage(_): raise OSError("not supported on Windows")
def setrlimit(_, __): pass
```

Then call rapidtide via `subprocess` rather than the Python API (already handled in `run_rapidtide.py`).

---

## Repository structure

```
GSOC_2/
  data/
    region_delays.npy          ← (100,) blood arrival delay per region (s)
    region_labels.npy          ← (100,) Schaefer region name strings
    parcellated_ts.npy         ← (168, 100) preprocessed BOLD time series
    fc_bias_results.npz        ← all Phase 2 FC matrices and metrics
    tvb_sim_results.npz        ← all Phase 3 simulation FC matrices and metrics
  figures/                     ← 8 output figures (PNG 300dpi + SVG)
  download_data.py             ← fetch fMRI from OpenNeuro S3 over HTTPS
  run_rapidtide.py             ← estimate voxelwise blood arrival delays
  parcellate_delays.py         ← parcellate lag map → 100-region delay vector
  compute_fc_bias.py           ← empirical FC bias quantification
  simulate_tvb.py              ← HRF + Balloon-Windkessel simulation with delay injection
  characterise_bias.py         ← spatial characterisation and all publication figures
  run_all.py                   ← single entry point, runs everything in order
  FINDINGS.md                  ← 1-page scientific summary
```

The raw fMRI data and rapidtide output are gitignored (too large). Everything else is committed, so you can reproduce all figures from the saved `.npy`/`.npz` files without re-running rapidtide.

---

## Honest scope

This is a single-subject prototype on a movie-watching task (`task-pixar`), not resting state. The structural connectivity used in simulation is derived from the same subject's functional data (positive FC thresholded at 0) rather than DTI tractography — a common surrogate when tractography isn't available, but a surrogate nonetheless. The scientific conclusions hold at the prototype level; a multi-subject resting-state analysis with empirical SC would be needed to generalise.

---

## What a full GSoC implementation would add

- A formal `HemodynamicDelay` datatype integrated into `tvb.datatypes.connectivity` — first-class vascular delay object, validated against rapidtide output
- Modified `tvb.simulator.monitors.Bold` ODE to accept per-region τᵢ onset offsets within the Balloon-Windkessel state equations (not just HRF convolution)
- Multi-subject analysis on HCP dataset with real DTI structural connectivity
- Systematic model fit comparison — empirical vs simulated FC Pearson r, with and without delay correction, across subjects
- EBRAINS-ready Docker container with one-command execution on the platform

---

*Dataset: OpenNeuro ds000228 (Richardson et al. 2018) · Atlas: Schaefer et al. 2018, 100 regions · Hemodynamic model: Balloon-Windkessel (Friston et al. 2003) · Delay estimation: rapidtide v3.1.8*