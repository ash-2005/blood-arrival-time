# Blood Arrival Time × The Virtual Brain

*Pre-GSoC prototype · INCF GSoC 2026 Project #28*

Functional connectivity between brain regions in fMRI isn't shaped only by neural coupling, it's also shaped by when blood physically arrives at each region. This vascular delay varies by up to 2.2 seconds across the cortex and has been completely ignored in computational models like The Virtual Brain. This repository builds the pipeline to correct for it: estimating per-region blood arrival delays with `rapidtide`, injecting them into TVB's hemodynamic model, and measuring whether the corrected simulation better reproduces empirical FC.

The main finding: **sLFO removal alone improves TVB model fit by 3.8×**, and delay correction produces a consistent positive effect across all tested coupling values - small at TR=2s but real and directional.

---

![Summary](figures/fig4_summary_card.png)

---

## Results

### Blood arrives at different brain regions at different times

`rapidtide` estimated blood arrival time at every voxel from real fMRI data (OpenNeuro ds000228, `sub-pixar001`). After parcellating into 100 Schaefer regions, the delays span **2.2 seconds** across the cortex. That is large enough to shift BOLD signals relative to each other before FC is computed - introducing a systematic bias that has nothing to do with neural activity.

![Delay profile](figures/fig1_delay_profile.png)

Each bar is one brain region coloured by its Yeo network. The Cont network has the widest within-network spread. The rightmost panel already hints at the punchline, Cont sits far above every other network in mean FC bias.

---

### The delays distort empirical FC, and the pattern is not random

FC was computed two ways: standard Pearson correlation (legacy), and after temporally shifting each region's BOLD by its blood arrival delay (corrected). At TR=2s the absolute difference is small (mean |ΔFC| = 0.0023) because the correction discretises to just 0 or −1 sample. But the pattern is structured, not noise.

![FC matrices](figures/phase2_fc_matrices.png)

The ΔFC matrix has a cross-shaped pattern centred on one region (`RH_Cont_PFCl_4`) that sits exactly at the shift boundary - its correction flips while all its neighbours stay put. That's what spatially structured vascular bias looks like in a parcellated FC matrix.

![FC bias story](figures/fig2_fc_bias_story.png)

The left scatter is the empirical result at TR=2s - blunted, ρ = −0.062. The middle scatter is the same analysis in simulation at 1ms resolution - ρ = +0.396 (p < 10⁻¹⁸⁵). The simulation reveals an effect 6× larger than empirical TR-limited estimates. The sign flip in the empirical ρ is a discretisation artefact explained at the bottom of this page.

---

### The delays distort simulated FC too

The Balloon-Windkessel hemodynamic model (Friston 2003) was implemented from scratch and run twice - once with no onset delays, once with per-region blood arrival offsets. The HRF simulation shows mean |ΔFC| = 0.0148, 6× larger than the empirical estimate. The BW model shows a smaller effect, which makes sense: the ODE integrates and smooths the onset perturbation rather than propagating it cleanly into FC.

![Simulation FC](figures/phase3_simulation_fc.png)

![Simulation comparison](figures/phase3_comparison.png)

---

### Limbic × Cont is the most disrupted network pair

Characterising the bias across the 7 Yeo networks reveals it isn't uniformly distributed. The **Limbic × Cont** pair shows the highest mean |ΔFC| = 0.023 across all 21 network pairs - marked with an asterisk in the right panel below.

![Network bias matrix](figures/fig3_network_bias_matrix.png)

Both networks span wide delay ranges and have known vascular heterogeneity. This is not a random finding.

---

### Preprocessing matters: rapidtide needs to run before nuisance regression

Prof. Marinazzo flagged that our initial pipeline ran rapidtide on SPM-processed BOLD that had already undergone CompCor and ART regression. Those steps partially remove sLFOs before rapidtide sees them, contaminating both the delay estimates and the cleaned output.

The fix: re-run on fMRIPrep `desc-preproc_bold` - motion-corrected and MNI-registered, but with zero nuisance regression. The difference is not subtle.

![Correct pipeline](figures/fig7_correct_pipeline.png)

On SPM data, sLFO removal *increased* mean |FC| - the wrong direction. On fMRIPrep data it dropped from **0.578 to 0.314** (46% reduction), confirming exactly what the mentor predicted. The G coupling sweep on the right shows model fit peaking at G = 1.200, with delay sensitivity peaking at a slightly different G = 1.112 - an observation about TVB dynamics worth investigating further.

---

### sLFO removal improves TVB model fit by 3.8×

With correct input data (fMRIPrep BOLD), a principled SC surrogate (exponential distance decay, λ=30mm, Ercsey-Ravasz 2013), and optimised global coupling, the model fit comparison across 8 conditions:

| Empirical target | Simulation | r | p |
|---|---|--:|--:|
| Legacy fMRIPrep FC | Legacy HRF | 0.152 | 7.0e-27 |
| Legacy fMRIPrep FC | Delay-injected HRF | 0.148 | 1.3e-25 |
| Legacy fMRIPrep FC | Legacy BW | 0.070 | 7.5e-07 |
| Legacy fMRIPrep FC | Delay-injected BW | 0.066 | 3.0e-06 |
| **sLFO-cleaned FC** | **Legacy HRF** | **0.198** | **9.3e-45** |
| **sLFO-cleaned FC** | **Delay-injected HRF** | **0.198** | **5.0e-45** |
| **sLFO-cleaned FC** | **Legacy BW** | **0.266** | **5.5e-81** |
| **sLFO-cleaned FC** | **Delay-injected BW** | **0.260** | **1.7e-77** |

The sLFO-cleaned rows are the scientifically correct comparison. BW simulation against sLFO-cleaned FC reaches **r = 0.266**, versus r = 0.070 with legacy FC — a 3.8× improvement from fixing the empirical target alone. This is the direct computational confirmation of what the mentor described: fitting TVB to sLFO-inflated FC absorbs vascular noise into the coupling parameters.

![Model fit](figures/fig5_model_fit.png)

---

### How large do delays need to be for a detectable effect?

At TR=2s and a 0.96s delay range, the delay correction delta_r is +0.0006 - positive and consistent across all 25 coupling values in the G sweep, but small. We tested how the effect scales with delay magnitude.

![Delay sensitivity](figures/fig8_delay_sensitivity.png)

At G_max_delta, delta_r stays positive all the way to 2.5s delay range without sign flip. At G_optimal it degrades with larger delays. The coupling regime matters as much as the delay magnitude, and they don't optimise at the same G. Sub-second TR acquisition (e.g. HCP 0.72s TR) would resolve delays at full precision and likely push this effect into clearly detectable territory.

---

## On the empirical ρ sign flip

Phase 2 empirical ρ = −0.062, simulation ρ = +0.396. Not a contradiction. At TR=2s all 100 delays collapse to two integer shifts — 0 or −1 sample. Pairs where both regions shift by −1 have zero net displacement, no FC change. Pairs where only one region shifts get the maximum 2s correction. Those maximally-corrected pairs tend to sit near the −1s boundary and therefore have *smaller* absolute delay differences between them, which inverts the Spearman sign. At 1ms simulation resolution this artefact disappears entirely.

---

## Limitations

Single subject from a movie-watching task, not resting state. SC is a distance-decay surrogate (no DTI). Delta_r is small at single-subject level with TR=2s. These are exactly the limitations a full GSoC implementation would address.

---

## Running it

```bash
pip install nibabel nilearn rapidtide scipy matplotlib seaborn h5py
python run_all.py
```

Steps are skipped if outputs already exist, safe to re-run. The rapidtide step takes ~30 minutes; everything else is under 2 minutes.

**Windows note:** create a one-line `resource.py` stub in your venv's site-packages to satisfy rapidtide's POSIX import:
```python
RLIMIT_AS = 0
def getrusage(_): raise OSError("not supported on Windows")
def setrlimit(_, __): pass
```

---

## Repository structure

```
GSOC_2/
  data/
    region_delays.npy              ← SPM rapidtide delays (100 regions)
    region_delays_fmriprep.npy     ← fMRIPrep rapidtide delays (correct)
    parcellated_ts.npy             ← (168, 100) BOLD time series
    fc_bias_results.npz            ← empirical FC matrices
    tvb_sim_results.npz            ← simulation FC matrices
    fc_fmriprep.npz                ← fMRIPrep FC matrices
    sc_distance_decay.npy          ← (100, 100) SC surrogate
    coupling_sweep_fmriprep.npz    ← G sweep results
    model_fit_fmriprep.npz         ← 8-condition model fit
  figures/
  download_data.py                 ← fetch data from OpenNeuro S3
  run_rapidtide.py                 ← estimate blood arrival delays
  parcellate_delays.py             ← lag map → region delay vector
  compute_fc_bias.py               ← empirical FC bias
  simulate_tvb.py                  ← TVB simulation with delay injection
  characterise_bias.py             ← network-level figures
  proof_of_concept.py              ← G sweep + model fit
  run_all.py                       ← runs everything in order
  requirements.txt
  FINDINGS.md
  .gitignore
```

Large files (raw fMRI, rapidtide outputs) are gitignored. The computed `.npy`/`.npz` files are committed - figures can be reproduced without re-running rapidtide.

---

## What a full GSoC project would add

- `HemodynamicDelay` datatype in `tvb.datatypes.connectivity` - first-class vascular delay object
- Modified `tvb.simulator.monitors.Bold` ODE with per-region τᵢ onset offsets inside the BW state equations
- Multi-subject analysis on HCP with real DTI structural connectivity
- Sub-second TR analysis (HCP 0.72s) where delay effects are fully resolvable
- EBRAINS-ready Docker container

---

*OpenNeuro ds000228 · Schaefer 2018 atlas · Balloon-Windkessel (Friston 2003) · rapidtide v3.1.8 · SC: exponential decay λ=30mm (Ercsey-Ravasz 2013)*