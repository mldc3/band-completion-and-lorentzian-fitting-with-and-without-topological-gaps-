# Theory and Background

## 1. Acoustic resonator chains as solid-state analogues

A one-dimensional chain of acoustic resonators can reproduce many of the mathematical structures that appear in elementary solid-state physics. In a crystal, atoms or orbitals provide localized degrees of freedom and the overlap between neighbouring orbitals produces bands. In the acoustic experiment, each cylinder plays the role of a localized resonator and each diaphragm controls how strongly sound couples from one cylinder to the next.

The analogy is useful because the experiment is macroscopic and tunable. Instead of changing microscopic orbital overlaps, the coupling is controlled by the aperture of the diaphragms. A larger aperture allows stronger transmission between neighbouring resonators, while a smaller aperture acts as a weaker link and increases reflection. This makes the system a natural platform for studying band formation, band gaps, dimerization, defects, and SSH-like topological modes.

The measured quantity is the acoustic transmission amplitude as a function of frequency. Peaks in this spectrum correspond to resonant modes of the coupled system. Once these peaks are identified, they can be organized into bands by assigning an effective wave number and grouping peaks separated by large frequency jumps.

## 2. Isolated cylindrical resonators

A single cylindrical acoustic cavity supports standing-wave resonances. In the simplest approximation, the resonant frequencies are

$$
f_n \simeq \frac{n c}{2L}, \qquad n = 1,2,3,\ldots
$$

where:

- $c$ is the speed of sound,
- $L$ is the effective length of the resonator,
- $n$ labels the longitudinal mode.

This formula explains why the length of the cylinder determines the approximate frequency region where bands appear. Large cylinders and medium cylinders do not merely change the visual geometry of the chain: they shift the local resonant frequencies of each site. In the tight-binding analogy, this is similar to changing the on-site energy of an atom.

In the raw experimental spectrum, an isolated resonance is well described by a Lorentzian line shape,

$$
L(f; A, f_0, \Gamma)
= A \frac{(\Gamma/2)^2}{(f-f_0)^2 + (\Gamma/2)^2},
$$

where $A$ is the resonance amplitude, $f_0$ is the resonance center, and $\Gamma$ is the full width at half maximum. A useful diagnostic is the quality factor

$$
Q = \frac{f_0}{\Gamma}.
$$

A large $Q$ indicates a sharp and well-defined resonance. A smaller $Q$ indicates broader modes, stronger losses, mode overlap, or experimental broadening.

## 3. From isolated resonators to acoustic bands

When several identical resonators are connected through diaphragms, the isolated resonances hybridize. Instead of one resonance per cylinder frequency, the chain develops several collective modes. For a chain of $N$ coupled resonators, one expects approximately $N$ resonant peaks per band, although finite resolution, losses, and mode overlap can hide some peaks.

A minimal tight-binding-like description writes the dispersion as

$$
\omega(k) = \omega_0 + 2J\cos(ka),
$$

where:

- $\omega_0$ is the isolated resonator frequency,
- $J$ is the effective coupling between neighbouring resonators,
- $a$ is the lattice spacing,
- $k$ is the wave number.

The bandwidth scales approximately with the coupling:

$$
\Delta \omega \sim 4|J|.
$$

Therefore, changing the diaphragm aperture changes the bandwidth. Larger diaphragms correspond to stronger coupling and wider bands. Smaller diaphragms correspond to weaker coupling, narrower bands, stronger localization, and often poorer transmission.

This explains a recurring trend in the experimental figures: the approximate position of a band is mostly set by the cylinder resonance, while the internal width and structure of the band are controlled by the diaphragm coupling.

## 4. The SSH model

The Su-Schrieffer-Heeger model is a one-dimensional tight-binding model with alternating couplings. Instead of a uniform hopping $J$, the chain alternates between two hoppings, often denoted $v$ and $w$:

$$
\cdots - v - w - v - w - v - w - \cdots
$$

In Bloch form, the SSH Hamiltonian can be written as

$$
H(k) =
\begin{pmatrix}
0 & v + w e^{-ika} \\
v + w e^{ika} & 0
\end{pmatrix}.
$$

The eigenvalues are

$$
E_\pm(k) = \pm \sqrt{v^2 + w^2 + 2vw\cos(ka)}.
$$

For an acoustic analogue, the absolute energy scale is replaced by frequency, and the two branches represent split acoustic bands or sub-bands. The important feature is the gap between branches. At the Brillouin-zone boundary, the size of the gap is controlled by the difference between the two couplings:

$$
\Delta \propto 2|v-w|.
$$

If $v=w$, the chain behaves like a uniform chain and the gap closes. If $v\neq w$, the alternating coupling opens a gap. This is the basic mechanism behind SSH-like band splitting.

## 5. Topological and trivial configurations

The SSH model has two dimerizations. In a finite chain, one arrangement is topologically trivial and the other can support edge or interface modes. The distinction is not simply whether a gap exists; both trivial and topological dimerized chains can have a gap. The distinction is whether the pattern of strong and weak bonds leaves an unpaired boundary or domain-wall state.

In an acoustic experiment, the topological interpretation must be adapted carefully. A localized peak inside or near a gap is a candidate for a topological mode only if it appears because of the coupling pattern and not merely because of a geometric defect, temperature shift, or random disorder.

A stronger interpretation is supported when:

1. alternating strong and weak couplings split the spectrum into sub-bands;
2. a central or boundary defect produces an isolated resonance inside the gap;
3. the spectrum remains approximately symmetric around the localized mode;
4. similar configurations reproduce the same feature;
5. ordinary defect cases do not produce the same robustness.

## 6. Mapping the SSH model onto the acoustic experiment

The mapping used throughout the project is:

$$
\text{cylinder} \leftrightarrow \text{site},
\qquad
\text{diaphragm} \leftrightarrow \text{hopping},
\qquad
\text{frequency} \leftrightarrow \text{energy}.
$$

Large, medium, small, 7 mm, and 4 mm diaphragms represent different coupling strengths. A qualitative ordering is

$$
J_{4\text{mm}} < J_{7\text{mm}} < J_{\text{small}} < J_{\text{medium}} < J_{\text{large}},
$$

although the exact numerical mapping depends on losses, geometry, and calibration.

Changing the cylinder size changes the local resonant frequency. This is closer to changing an on-site term than changing a hopping term. Therefore, replacing one cylinder by a different size is a defect problem, not a pure SSH dimerization problem.

## 7. Disorder, finite-size effects, and missing peaks

The experiment is finite, lossy, and measured with finite spectral resolution. This has several important consequences.

First, the number of peaks per band is ideally related to the number of cylinders, but this is not always visible experimentally. Peaks can merge, broaden, or fall below the detection threshold.

Second, higher-frequency bands often show weaker or broader peaks. This can happen because losses increase, resonances overlap more strongly, and the measurement system has finite sensitivity.

Third, a finite chain can hybridize edge states. Two modes localized at opposite ends are not perfectly independent unless the system is sufficiently long and the gap is large enough. If the chain is short, the two edge modes may split, move away from the center of the gap, or become hard to distinguish.

Fourth, random or chaotic diaphragm sequences can generate mini-gaps and localized-looking modes without being topological. This is why the project separates three classes of phenomena:

- ordinary band formation in periodic chains,
- disorder-induced band fragmentation,
- SSH-like gap states associated with alternating couplings and defects.

## 8. Why Lorentzian fitting is used

The raw spectra are not just collections of points. Each resonance has a finite width and overlaps with neighbouring resonances. Lorentzian fitting gives a more physically meaningful representation than simply reading the maximum sample value.

The fitted center $f_0$ is used as the resonance frequency. The width $\Gamma$ provides information about loss and mode definition. The amplitude $A$ indicates how strongly that mode transmits through the chain. Together, these parameters allow the script to reconstruct bands more reliably than a purely visual analysis.

For multiple resonances, the model is a sum of Lorentzians:

$$
A_{\text{model}}(f)=\sum_i A_i\frac{(\Gamma_i/2)^2}{(f-f_i)^2+(\Gamma_i/2)^2}.
$$

The scripts use this model to approximate the measured transmission curve and to refine the peak list used in the band diagrams.

## 9. Why interpolation and extrapolation are needed

Because some expected modes are missing, the band diagrams are incomplete if only directly detected peaks are plotted. The scripts therefore complete bands by inserting missing values at physically expected positions.

The most important constraint is that each band should contain approximately one mode per cylinder. If a band contains fewer peaks than expected, the missing entries are represented by `None` values and later filled using local searches or interpolation.

PCHIP interpolation is useful because it preserves the shape of the data better than a high-degree polynomial fit. It avoids strong artificial oscillations and is therefore appropriate for completing smooth band trends when the number of missing points is moderate.

The completed points should not be treated as direct measurements. In the figures, measured and interpolated points should be visually distinguished.

## 10. Interpretation limits

The project is strongest when presented as an experimental-data reconstruction and physical-interpretation pipeline. It should not claim that every gap feature is topological. Instead, the correct message is:

> The code reconstructs acoustic band structures from incomplete spectra and uses SSH-inspired theory to classify which configurations are consistent with periodic, disordered, trivial-defect, or topological-defect behaviour.

That careful framing makes the repository more credible and more useful for a technical audience.
