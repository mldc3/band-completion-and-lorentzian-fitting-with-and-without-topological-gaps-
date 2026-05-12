# Methods and Algorithms

This document explains the computational workflow used to transform raw acoustic spectra into reconstructed band diagrams.

## 1. Input data

The scripts assume experimental data files with two numerical columns:

```text
frequency amplitude
```

The first column is the excitation or measurement frequency, usually in hertz. The second column is the measured acoustic amplitude. Each experimental case corresponds to a different physical chain: a specific number of cylinders, cylinder sizes, diaphragm apertures, and possible defects or temperature variations.

Before running the analysis, the user chooses several case-dependent parameters:

| Parameter | Meaning |
|---|---|
| `file_path` or `data_path` | Experimental `.dat` file to analyze. |
| `frequency_min_plot`, `frequency_max_plot` | Frequency window used for detection and plotting. |
| `num_cylinders` | Expected number of modes per band. |
| `L` or `system_length` | Effective chain length used to assign wave numbers. |
| `amplitude_threshold` | Minimum amplitude for accepting a peak. |
| `frequency_gap_threshold` | Frequency jump used to separate bands or sub-bands. |
| `peak_distance` | Minimum spacing between candidate peaks. |

These parameters are intentionally exposed because the experimental cases are not identical. A future version should move them to external configuration files.

## 2. Peak detection

The first computational step is peak detection. The scripts use `scipy.signal.find_peaks`, usually with a prominence threshold. This identifies local maxima that stand out from neighbouring data points.

The basic idea is:

```python
peaks, props = find_peaks(amplitude, prominence=prominence_threshold)
filtered_peaks = [i for i in peaks if amplitude[i] > amplitude_threshold]
```

This produces a first list of candidate resonances. The frequency and amplitude of each peak are then extracted:

```python
frequency_peaks = frequency[filtered_peaks]
amplitude_peaks = amplitude[filtered_peaks]
```

The purpose of the amplitude threshold is to remove weak fluctuations and noise. The purpose of the prominence threshold is to avoid selecting points that are local maxima only because of small numerical noise.

## 3. Lorentzian fitting

Each acoustic resonance is approximated as a Lorentzian:

$$
L(f; A, f_0, \Gamma)
= A \frac{(\Gamma/2)^2}{(f-f_0)^2 + (\Gamma/2)^2}.
$$

For a spectrum with many peaks, the fitting function is a sum of Lorentzians:

$$
A_{\mathrm{fit}}(f)=\sum_{i=1}^{N_p} L_i(f; A_i, f_i, \Gamma_i).
$$

The code builds an initial parameter list containing amplitude, center, and width for each detected peak. Peak widths are estimated with `scipy.signal.peak_widths`. The nonlinear optimization is performed using `scipy.optimize.curve_fit`.

The fitted parameters allow the script to extract:

- peak center frequencies,
- fitted amplitudes,
- linewidths,
- approximate quality factors $Q=f_0/\Gamma$.

This is useful because raw peak positions can be distorted by sampling resolution or overlapping resonances.

## 4. Assigning wave number

Once the peak frequencies are known, the scripts assign a discrete wave number using

$$
k_i = \frac{i\pi}{L},
$$

where $i$ is the index of the mode and $L$ is the effective length of the chain. This is a finite-chain approximation: it is not a full Bloch analysis of an infinite system, but it provides a consistent horizontal axis for visualizing the measured modes as a band diagram.

The code then mirrors the data around $k=0$ to obtain a symmetric band representation from negative to positive wave number.

## 5. Band separation

The peak list is divided into bands by detecting large frequency jumps. If two consecutive peak frequencies differ by more than a selected threshold, the midpoint is treated as a boundary between bands or sub-bands.

In simplified form:

```python
for i in range(len(frequency_peaks)-1):
    jump = abs(frequency_peaks[i+1] - frequency_peaks[i])
    if jump > frequency_gap_threshold:
        boundary = (k[i] + k[i+1]) / 2
        band_boundaries.append(boundary)
```

This reflects the physical fact that acoustic bands appear as clusters of resonances separated by regions of low transmission.

## 6. Band normalization

For a chain with `num_cylinders = N`, each band is expected to contain approximately $N$ modes. However, the experiment often misses peaks because of losses, overlap, weak transmission, or limited resolution.

To maintain a consistent band representation, each band is normalized to length $N$. If fewer than $N$ peaks are present, the code inserts `None` values at plausible missing positions. These missing entries are later completed.

This is important for plotting and for comparing different cases, because all bands must have compatible lengths before they can be reflected or interpolated.

## 7. Missing-mode completion

The project uses two complementary strategies to complete missing band values.

### 7.1 Local search

When the approximate location of a missing mode is known, the script searches locally in the raw spectrum. Depending on the situation, it looks for:

- a clear peak between two known resonances,
- a peak before the first known resonance,
- a peak after the last known resonance,
- a soft shoulder or inflection point when two resonances overlap.

This is particularly useful in topological or disordered configurations, where some peaks are weak or partially merged.

### 7.2 PCHIP interpolation

When a band trend is smooth but one or more points are missing, the code uses a Piecewise Cubic Hermite Interpolating Polynomial (`PchipInterpolator`). PCHIP is preferred because it tends to preserve monotonicity and avoids the artificial oscillations associated with high-order polynomial interpolation.

Interpolated points are not treated as experimental measurements. They are estimates used to visualize the likely band structure.

## 8. Topological-gap workflow

The script for topological cases contains additional logic because the expected structure can include sub-bands and central localized modes. In these cases, the user may define an expected band shape, for example using lists containing `1` for expected observed modes and `None` for missing modes.

This manual prior information encodes physical knowledge of the expected band organization. The algorithm then fills the structure with measured peaks, searches for missing points, and reconstructs the full diagram.

This is not a weakness; it reflects the reality of experimental data analysis. Completely automatic peak assignment is difficult when peaks merge, disappear, or change shape. The value of the script is that it makes the assumptions explicit and reproducible.

## 9. SSH comparison

After reconstructing the bands, the results can be compared with SSH-inspired dispersion relations. The simplest SSH-like expression has two branches:

$$
\omega_\pm(k)=\omega_0 \pm \sqrt{J_1^2 + J_2^2 + 2J_1J_2\cos(ka)}.
$$

The parameters $J_1$ and $J_2$ represent effective strong and weak couplings. In the experiment, they are controlled by diaphragm apertures. The comparison is qualitative unless the acoustic couplings have been independently calibrated.

## 10. Validation checks

A reconstructed band diagram is more credible when several checks pass:

1. The number of modes per band is consistent with the number of cylinders.
2. Band positions are consistent with cylinder size.
3. Bandwidth changes consistently with diaphragm aperture.
4. Missing points are visually plausible in the raw spectrum.
5. Interpolated points are marked separately from measured points.
6. Topological interpretations are compared against non-topological defect cases.
7. Thermal and geometric perturbations are not mistaken for protected modes.

## 11. Limitations

The present scripts are valuable research prototypes but should not be presented as fully automated production software. Main limitations:

- Parameters are hard-coded per case.
- Input file names are edited directly in the script.
- Several analysis choices require physical judgment.
- Some functions use global variables.
- Plot saving is not yet standardized.
- Uncertainty estimates are not propagated into the final band diagrams.

These limitations are normal for an exploratory internship project. They also provide a clear roadmap for future improvements.
