# Acoustic SSH Band Reconstruction from Experimental Resonator Data

This repository studies one-dimensional acoustic resonator chains as experimentally accessible analogues of tight-binding and Su-Schrieffer-Heeger (SSH) systems. The project takes raw frequency-amplitude measurements from coupled cylindrical resonators, detects resonant peaks, fits Lorentzian profiles, reconstructs acoustic band diagrams, completes missing spectral points, and compares periodic, disordered, trivial, and topological configurations.

The central idea is simple: each cylinder behaves like an acoustic resonator, while each diaphragm controls the coupling between neighbouring resonators. By changing the diameter, order, symmetry, temperature, and defect structure of the chain, the measured spectrum changes from isolated resonances to bands, mini-gaps, and localized mid-gap-like modes.

## Why this project is interesting

This is not just a plotting exercise. The code implements a full experimental-data workflow:

- Raw frequency-amplitude spectra are processed automatically.
- Resonances are detected using peak prominence, amplitude thresholds, and local searches.
- Lorentzian fitting is used to recover peak centers, widths, and quality factors.
- Detected resonances are grouped into acoustic bands and sub-bands.
- Missing modes are completed using physically constrained interpolation/extrapolation.
- Band diagrams are compared against tight-binding and SSH-inspired expectations.
- Topological and non-topological configurations are discussed carefully rather than overclaimed.

The result is a portfolio-style computational physics project connecting experimental data analysis, numerical methods, acoustic metamaterials, and topological band theory.

## Repository guide

| Section | Purpose |
|---|---|
| `docs/theory_and_background.md` | Full theoretical explanation: acoustic resonators, tight-binding bands, SSH model, gaps, localized modes, and finite-size effects. |
| `docs/methods_and_algorithms.md` | Description of the complete data-analysis pipeline used by the scripts. |
| `docs/results_and_discussion.md` | Detailed discussion of representative figures and physical trends. |
| `docs/implementation_notes.md` | Explanation of what each script does, expected inputs, parameters, limitations, and future refactoring ideas. |
| `assets/figures/` | Selected visual results from the experimental report. |

## Numerical workflow

The workflow implemented in the repository can be summarized as:

```text
experimental .dat file
        |
        v
frequency-amplitude spectrum
        |
        v
peak detection with scipy.signal.find_peaks
        |
        v
Lorentzian fitting and peak-parameter extraction
        |
        v
band grouping using frequency gaps and expected number of modes
        |
        v
missing-mode completion with local searches and PCHIP interpolation
        |
        v
reconstructed acoustic band diagram
        |
        v
physical interpretation using tight-binding and SSH analogies
```

## The physical system

The experimental system consists of a speaker, a one-dimensional chain of coupled cylindrical resonators, diaphragms between neighbouring cylinders, and a microphone measuring the transmitted acoustic response. In the solid-state analogy:

| Acoustic experiment | Tight-binding / SSH analogy |
|---|---|
| Cylinder | Lattice site / resonator / artificial atom |
| Cylinder resonance | On-site energy or local mode frequency |
| Diaphragm aperture | Coupling / hopping amplitude |
| Large diaphragm | Stronger coupling |
| Small diaphragm | Weaker coupling |
| Alternating diaphragms | Dimerized SSH chain |
| Defect or domain wall | Localized mode candidate |
| Frequency gap | Acoustic band gap / mini-gap |
| Peak in gap | Localized defect/topological mode candidate |

## Selected results

### Disordered large-cylinder configuration

![Disordered 8-large-cylinder configuration](assets/figures/disorder/fig_disorder_8_large_config1.png)

This case shows how an irregular sequence of diaphragm apertures modifies the acoustic spectrum. The measured resonances still cluster into bands, but the band diagram is less regular than in a homogeneous chain. This makes it useful as a bridge between clean periodic systems and deliberately engineered topological configurations.

### Defect introduced by one medium cylinder

![15 large cylinders and one medium cylinder](assets/figures/defects/fig_defect_15_large_1_medium.png)

Replacing one large cylinder with a medium one changes the local resonance frequency. This introduces an on-site defect rather than a purely SSH-like hopping defect. The spectrum is therefore distorted, but the interpretation must be different from a protected topological mode: here the localized behaviour is mainly caused by a local geometric perturbation.

### Topological 24-medium-cylinder configuration

![Topological 24-medium-cylinder configuration](assets/figures/topological/fig_topological_24_medium_large_medium.png)

The 24-cylinder topological configurations are among the clearest visual examples because the larger system size gives a denser set of modes and a more recognizable band structure. The red points in the band diagrams mark candidate topological or gap-related points, and the corresponding spectra show how alternating strong and weak couplings split bands into sub-bands.

### Thermal perturbation

![Temperature-dependent topological configuration](assets/figures/temperature/fig_temperature_16_medium_cold_topological.png)

Temperature changes the speed of sound and therefore shifts local resonance frequencies. In the experiment this acts as an additional perturbation that can distort the band structure. This case is useful because it shows that not every isolated spectral feature should be interpreted as topological: thermal and geometric disorder can also produce localized or anomalous peaks.

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the scripts

The original scripts are configured by editing parameters directly in the script, including the input `.dat` file, frequency range, number of cylinders, system length, amplitude threshold, and frequency-gap threshold.

Example:

```bash
python src/band_analysis_with_topological_gaps.py
python src/frequency_band_completion_and_lorentzian_fitting.py
```

The scripts expect experimental `.dat` files with two columns:

```text
frequency amplitude
```

A future improvement would be to move all case-dependent parameters into YAML or JSON configuration files so that the same pipeline can be reused without editing the source code.

## Scientific interpretation

The repository uses SSH language because the experiment is designed around a one-dimensional chain with tunable nearest-neighbour couplings. However, the interpretation is intentionally careful. A peak inside or near a gap is not automatically a topological mode. The strongest evidence comes from a combination of observations:

- a gap or mini-gap created by alternating couplings,
- a localized mid-gap-like resonance associated with a domain wall or defect,
- approximate symmetry of the spectrum around the localized state,
- robustness under moderate perturbations,
- reproducibility across similar configurations,
- distinction from ordinary geometric or thermal defects.

The project therefore presents the acoustic chains as SSH-inspired experimental analogues and focuses on the computational reconstruction and interpretation of their band structures.

## Suggested final repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── band_analysis_with_topological_gaps.py
│   └── frequency_band_completion_and_lorentzian_fitting.py
├── docs/
│   ├── theory_and_background.md
│   ├── methods_and_algorithms.md
│   ├── results_and_discussion.md
│   └── implementation_notes.md
└── assets/
    ├── ASSETS_INDEX.md
    └── figures/
        ├── comparisons/
        ├── defects/
        ├── disorder/
        ├── topological/
        ├── temperature/
        └── weak_coupling/
```

## Project scope

This repository is best understood as a polished research-internship project: it does not claim a final experimental discovery, but it demonstrates the construction of a serious analysis pipeline for incomplete experimental spectra in a physically motivated topological acoustic system.
