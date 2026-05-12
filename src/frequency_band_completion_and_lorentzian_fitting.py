import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, argrelextrema
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
import string


'''
FIRST PART
In this section, we recover the function describing amplitude as a function of frequency from experimental data.
To do this, we take the peaks present in the signal and fit them with Lorentzian functions, which are typical for describing resonances.
Finally, we filter the peaks of interest (by amplitude and frequency), extract their characteristic parameters,
and visualize the original curve along with the sum of Lorentzians fitted to the experimental data.
'''

file_path = "Normal_3(cgrandes,dgrandes).dat"  # Name of the file to analyze
data = np.loadtxt(file_path)
full_frequency = data[:, 0]
full_amplitude = data[:, 1]

# Set the range we want to plot and analyze
min_plot_frequency = 2000
max_plot_frequency = 10000
num = np.argmin(np.abs(full_frequency - min_plot_frequency))

num_cylinders = 8            # Helps determine how many peaks to find per band
amplitude_threshold = 0.2      # Peaks below this amplitude are ignored
case_name = "8 large cylinders with large diaphragms"
system_length = 0.075 * num_cylinders  # System length, adjust according to system size
frequency_jump_threshold = 500       # Maximum frequency between consecutive peaks to consider same band
k_threshold = 2

# Defines a resonance centered at nu0 with width gamma and maximum amplitude A
# This will allow us to recover the original function just using the peaks
def lorentzian(nu, A, nu0, gamma):
    return A * (gamma / 2)**2 / ((nu - nu0)**2 + (gamma / 2)**2)

# Takes a concatenated list of parameters [A1, mu1, g1, A2, mu2, g2, ...] and sums the curves of each peak
# Used to fit all peaks in the signal simultaneously
def multi_lorentzian(nu, *params):
    model = np.zeros_like(nu)
    for i in range(len(params)//3):
        A, mu, g = params[3*i:3*i+3]
        model += lorentzian(nu, A, mu, g)
    return model

# Detect peaks in the signal using find_peaks from scipy.signal
# Returns indices of prominent local peaks with certain relative height
peaks, properties = find_peaks(full_amplitude, prominence=0.05)

# Filter peaks and keep only those with amplitude above threshold
filtered_peaks = [i for i in peaks if full_amplitude[i] > amplitude_threshold]
peaks = filtered_peaks

# Extract frequency and amplitude corresponding to filtered peaks
peak_frequencies = full_frequency[filtered_peaks]
peak_amplitudes = full_amplitude[filtered_peaks]

# Calculate full width at half maximum (FWHM) of each peak
# dx is the frequency resolution of the sampling
widths = peak_widths(full_amplitude, filtered_peaks, rel_height=0.5)[0]
dx = full_frequency[1] - full_frequency[0]
fwhm_initial = widths * dx

# Define initial parameter vector p0 for fitting: height, center, width for each peak
# Also define lower and upper bounds to guide the fitting (avoid non-physical values)
gamma_min = 10  # Hz
p0, lower_bounds, upper_bounds = [], [], []

for H, mu, fwhm in zip(peak_amplitudes, peak_frequencies, fwhm_initial):
    g0 = max(fwhm, gamma_min)
    p0 += [H, mu, g0]
    lower_bounds += [0, mu - 3*g0, gamma_min/2]
    upper_bounds += [np.inf, mu + 3*g0, 2*g0]

# Perform nonlinear fitting (curve_fit) using the sum of Lorentzians
# Finds parameters that best describe the complete signal
fitted_params, _ = curve_fit(multi_lorentzian, full_frequency, full_amplitude,
                             p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=30000)

# Extract fitted parameters for each Lorentzian (height, center, width)
# Filter those that fall within the range of interest
A_fit = fitted_params[0::3]
mu_fit = fitted_params[1::3]
gamma_fit = fitted_params[2::3]

mask_peaks = (mu_fit >= min_plot_frequency) & (mu_fit <= max_plot_frequency)
selected_indices = np.where(mask_peaks)[0]

selected_frequencies = peak_frequencies[selected_indices]
selected_amplitudes = peak_amplitudes[selected_indices]

peak_frequencies = selected_frequencies
peak_amplitudes = selected_amplitudes

# Display selected peaks and compute quality factor Q = mu / gamma for each peak
for i, idx in enumerate(selected_indices, 1):
    A = A_fit[idx]
    mu = mu_fit[idx]
    g = gamma_fit[idx]
    Q = mu / g
    #print(f"Peak {i}: amplitude={selected_amplitudes[i-1]:.2f}, mu={mu:.2f} Hz, gamma={g:.2f}, Q={Q:.2f}")

# Extract segment of the signal in the defined plotting range
mask_plot = (full_frequency >= min_plot_frequency) & (full_frequency <= max_plot_frequency)
plot_frequency = full_frequency[mask_plot]
plot_amplitude = full_amplitude[mask_plot]

# Evaluate fitted curve in the same range
fitted_curve = multi_lorentzian(plot_frequency, *fitted_params)

plt.figure(figsize=(10, 6))
plt.plot(plot_frequency, plot_amplitude, label='Data', color='cyan')
plt.plot(plot_frequency, fitted_curve, '--', label='Lorentzian Fit', color='darkblue')

# Scatter filtered original peaks, keeping their actual height
plt.scatter(selected_frequencies, selected_amplitudes, color='royalblue', label='Selected Peaks')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title(f'Lorentzian Peak Fit using find_peaks: {case_name}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


'''
SECOND PART
In this section, we identify Brillouin zones and construct the band diagram.
We use the previously selected peaks to:
1. Calculate the wave-number associated with each peak based on its order (mode).
2. Identify large frequency jumps that indicate potential bandgap boundaries.
3. Group peaks into bands (allowed zones).
4. Normalize bands to have the same length.
5. Create complete symmetric bands for more realistic plotting.
6. Generate a band diagram plotting frequency vs wave-number.
'''

# ################################ BRILLOUIN ZONES ################################
# Calculate the wave-number for each peak as: k_i = i * pi / L, where L is the total system length
wave_numbers = [i * np.pi / system_length for i in range(len(peak_frequencies))]

# Calculate frequency jumps and k-boundaries
k_jumps = []
for i in range(len(peak_frequencies)-1):
    freq_jump = abs(peak_frequencies[i+1] - peak_frequencies[i])
    if freq_jump > frequency_jump_threshold:
        k_boundary = (wave_numbers[i] + wave_numbers[i+1]) / 2
        k_jumps.append(k_boundary)

# Build a list of Brillouin zone boundaries: start at first k, followed by detected jumps, end at last k
boundaries = [wave_numbers[0]] + k_jumps + [wave_numbers[-1]]
num_zones = len(boundaries) - 1

# Group peaks within each band using defined k-boundaries
bands = []
for i in range(num_zones):
    k_start, k_end = boundaries[i], boundaries[i+1]
    band = [f for k, f in zip(wave_numbers, peak_frequencies) if k_start <= k <= k_end]
    bands.append(band)
    if band:
        print(f"Band {i+1}: min={min(band):.2f}Hz, max={max(band):.2f}Hz, range={max(band)-min(band):.2f}Hz")

# Calculate minimum distance between adjacent bands (bandgap estimation)
min_band_distances = []
for i in range(num_zones-1):
    b1, b2 = bands[i], bands[i+1]
    if b1 and b2:
        dmin = min(abs(f1-f2) for f1 in b1 for f2 in b2)
        min_band_distances.append(dmin)
        #print(f"Min distance band {i+1}-{i+2}: {dmin:.2f}Hz")

# Normalize the length of each band to have num_cylinders elements for visual consistency
normalized_bands = []
for band in bands:
    diff = num_cylinders - len(band)
    b = list(band)
    # Alternate appending/prepending None
    for j in range(diff):
        if j % 2 == 0:
            b.append(None)
        else:
            b.insert(0, None)
    normalized_bands.append(b[:num_cylinders])

# Construct a symmetric band around k=0 by reflecting the band to the negative k-axis
symmetric_bands = []
for i, band in enumerate(normalized_bands):
    band_rev = band[::-1]
    if i % 2 == 0:
        full_band = band_rev[:-1] + band
    else:
        full_band = band + band_rev[1:]
    symmetric_bands.append(full_band)
    globals()[f"symmetric_band{i}"] = np.array(full_band, dtype=object)

# Generate a full symmetric wave-number axis from -k_max to k_max for plotting
k_full_len = len(symmetric_bands[0])
wave_numbers_plot = [i * np.pi / system_length for i in range(num_cylinders)]
wave_numbers_full = np.linspace(-max(wave_numbers_plot), max(wave_numbers_plot), 2*num_cylinders - 1)

plt.figure(figsize=(8,6))
for i, band in enumerate(symmetric_bands):
    plt.plot(wave_numbers_full, band, 'o', label=f'Band {i+1}')
plt.xlabel('Wave-number k')
plt.ylabel('Frequency (Hz)')
plt.title(f'Bands using only find_peaks: {case_name}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.tight_layout()
plt.show()


################################ INTERPOLATION ################################
'''
THIRD PART
In this section, we fill missing values in the previously obtained frequency bands by extrapolation.
This allows us to:
- Construct complete and smooth bands
- Compare obtained vs interpolated values
- Visualize a more continuous and symmetric band diagram, useful when some modes were not experimentally detected.
'''

# Extract the original frequency bands (by Brillouin zones) into two lists
bands_original = []          # stores obtained values
bands_interpolated = []      # will be modified via interpolation

for i in range(len(boundaries)-1):
    k_start, k_end = boundaries[i], boundaries[i+1]
    band_i = [f for k, f in zip(wave_numbers, peak_frequencies) if k_start <= k <= k_end]
    band_i_interp = band_i.copy()
    bands_original.append(band_i)
    bands_interpolated.append(band_i_interp)

# Adjust length of each band by adding None if values are missing
target_length = num_cylinders
for i in range(len(bands_original)):
    band_orig = bands_original[i]
    band_interp = bands_interpolated[i]
    diff = target_length - len(band_orig)
    for j in range(diff):
        if j % 2 == 0:
            band_orig.append(None)
            band_interp.append(None)
        else:
            band_orig.insert(0, None)
            band_interp.insert(0, None)
    bands_original[i] = band_orig[:target_length]
    bands_interpolated[i] = band_interp[:target_length]

# Separate known and missing indices (None)
for i, band_interp in enumerate(bands_interpolated):
    known_x = [j for j, v in enumerate(band_interp) if v is not None]
    known_y = [v for v in band_interp if v is not None]

    # Interpolate using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    if len(known_x) >= 2:
        try:
            interp = PchipInterpolator(known_x, known_y, extrapolate=True)
            bands_interpolated[i] = [float(interp(j)) if v is None else v
                                     for j, v in enumerate(band_interp)]
        except:
            bands_interpolated[i] = band_interp
    else:
        bands_interpolated[i] = band_interp

# Generate a symmetric version of each interpolated band for plotting -k to k
for i, band_interp in enumerate(bands_interpolated):
    band_rev = band_interp[::-1]
    band_rev.pop()
    if i % 2 == 0:
        full_band = band_rev + band_interp
    else:
        full_band = band_interp + band_rev
    globals()[f"interpolated_band_full{i}"] = np.array(full_band, dtype=object)

# Construct k-axis corresponding to extended (reflected) band length
k_initial = len(wave_numbers) - (2 * target_length - 1)
wave_numbers_final = wave_numbers[k_initial:]

colors = plt.cm.tab10.colors

plt.figure(figsize=(10, 6))
plt.title(f"Bands with extrapolation: {case_name}")
plt.xlabel("Wave-number k (1/mm)")
plt.ylabel("Frequency (Hz)")

# Plot all bands
band_labels_set = set()
for i in range(len(bands_original)):
    color = colors[i % len(colors)]
    band_orig = bands_original[i]
    band_interp = bands_interpolated[i]

    for j in range(target_length):
        val_orig = band_orig[j]
        val_interp = band_interp[j]

        if i % 2 == 0:
            idx_left = target_length - 1 - j
            idx_right = target_length - 1 + j
        else:
            idx_left = j
            idx_right = 2 * target_length - 2 - j

        if 0 <= idx_left < len(wave_numbers_full):
            if val_orig is not None:
                if f"Band {i+1}" not in band_labels_set:
                    plt.plot(wave_numbers_full[idx_left], val_orig, 'o', color=color, label=f"Band {i+1}")
                    band_labels_set.add(f"Band {i+1}")
                else:
                    plt.plot(wave_numbers_full[idx_left], val_orig, 'o', color=color)
            elif val_interp is not None:
                plt.plot(wave_numbers_full[idx_left], val_interp, '*', color=color, markersize=10)

        if 0 <= idx_right < len(wave_numbers_full):
            if val_orig is not None:
                plt.plot(wave_numbers_full[idx_right], val_orig, 'o', color=color)
            elif val_interp is not None:
                plt.plot(wave_numbers_full[idx_right], val_interp, '*', color=color, markersize=10)

# Define legend elements for measured vs interpolated data
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', linestyle='None', label='Measured Value'),
    Line2D([0], [0], marker='*', color='gray', markersize=10, linestyle='None', label='Interpolated Value')
]
for i in range(len(bands)):
    color = colors[i % len(colors)]
    legend_elements.append(Line2D([0], [0], marker='o', color=color, linestyle='None', label=f'Band {i+1}'))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()




################# 1st plot but with new points #################

bands = []
print("Checking that bands are empty", bands)
for i in range(num_zones):
    k0, k1 = boundaries[i], boundaries[i+1]
    band = [f for k, f in zip(wave_numbers, peak_frequencies) if k0 <= k <= k1]
    bands.append(band)
    if band:
        print(f"Band {i+1}: min={min(band):.2f}Hz, max={max(band):.2f}Hz, range={max(band)-min(band):.2f}Hz")

'''
This function analyzes each band to count how many real values it contains,
and then fills each band with None at the edges if points are missing. 
We do this because we need all bands to have the same length as num_cylinders,
so that they are aligned visually. Steps:
1. Iterate through each band and count real (non-None) values.
2. Calculate how many points are missing to reach 'num_cylinders':
   - If none are missing, save the band as is.
   - If an even number is missing, add half on the left and half on the right (symmetric).
   - If an odd number is missing, create two versions: one with more None on the left, one with more on the right.
Finally, all completed bands (and possible versions) are stored in a global variable
'completed_bands_possible'.
'''

def analyze_and_complete_bands(bands, num_cylinders):
    globals()['completed_bands_possible'] = []  # Initialize global list for all possible completed bands

    for i, band in enumerate(bands):
        current_band = band.copy()
        missing = num_cylinders - len([x for x in current_band if x is not None])

        if missing == 0:
            globals()['completed_bands_possible'].append([current_band])  # Only one possibility
            continue

        # Even case: same number of None on each side
        if missing % 2 == 0:
            n = missing // 2
            new_band = [None]*n + current_band + [None]*n
            globals()['completed_bands_possible'].append([new_band])

        # Odd case: two versions (more None left or more None right)
        else:
            n1 = (missing + 1) // 2
            n2 = (missing - 1) // 2
            possibility_1 = [None]*n1 + current_band + [None]*n2
            possibility_2 = [None]*n2 + current_band + [None]*n1
            globals()['completed_bands_possible'].append([possibility_1, possibility_2])

    return globals()['completed_bands_possible']

bands_possible = analyze_and_complete_bands(bands, num_cylinders)
print("Filled bands:", bands_possible)


def extrapolate_bands(bands, num_cylinders):
    '''
    Extrapolates aligned bands (with None at edges) and returns for each band:
      1. New interpolated values (without None)
      2. Full symmetric band
      3. Interpolated band (original band with None replaced)
      4. Reflected bands for symmetry
    '''
    results = []
    target_length = num_cylinders

    for i, band in enumerate(bands):
        known_x = [j for j, v in enumerate(band) if v is not None]
        known_y = [v for v in band if v is not None]
        unknown_x = [j for j, v in enumerate(band) if v is None]
        print("Unknown indices:", unknown_x)
        print("Known indices:", known_x)

        if len(known_x) >= 2:
            try:
                interp = PchipInterpolator(known_x, known_y, extrapolate=True)
                band_interp = [float(interp(j)) if v is None else v for j, v in enumerate(band)]
                new_values = [float(interp(j)) for j in unknown_x]
            except Exception as e:
                band_interp = band.copy()
                new_values = []
        else:
            band_interp = band.copy()
            new_values = []

        band_rev = band_interp[::-1]
        band_rev.pop()  # Remove central point to avoid duplication

        if i % 2 == 0:
            full_band = band_rev + band_interp
        else:
            full_band = band_interp + band_rev

        results.append([new_values, full_band, band_interp])
    return results

results = []
new_bands = []
full_bands = []
interpolated_bands = []

for band_options in bands_possible:
    band = band_options[0]  # Choose first version if multiple

    result = extrapolate_bands([band], num_cylinders)[0]
    new_vals, full_band, band_interp = result

    results.append([new_vals, full_band, band_interp])
    new_bands.append(new_vals)
    full_bands.append(full_band)
    interpolated_bands.append(band_interp)

new_bands1 = new_bands

for i, (new_points, full_band, band_interp) in enumerate(results):
    print(f"\n--- Band {i+1} ---")
    print("Extrapolated new points:", new_points)
    print("Full band:", full_band)
    print("Interpolated band:", band_interp)

    globals()[f"band{i+1}_new"] = new_points
    globals()[f"band{i+1}_full"] = full_band
    globals()[f"band{i+1}_interpolated"] = band_interp


def adjust_extrapolated_values(band_interp, new_extrapolated):
    """
    Adjusts extrapolated points in an interpolated band to match realistic experimental values.
    Searches for the closest experimental frequency with high amplitude or a smooth reflection.
    """
    final_band = band_interp.copy()

    for idx, val in enumerate(band_interp):
        if not any(np.isclose(val, new_val, atol=1e-3) for new_val in new_extrapolated):
            continue

        # Find the closest real value in the experimental data
        real_val = min(plot_frequency, key=lambda f: abs(f - val))
        val = real_val

        # Distances to neighboring points
        neighbors = []
        if idx > 0 and band_interp[idx - 1] is not None:
            neighbors.append(abs(val - band_interp[idx - 1]))
        if idx < len(band_interp) - 1 and band_interp[idx + 1] is not None:
            neighbors.append(abs(val - band_interp[idx + 1]))
        if not neighbors:
            continue

        delta = min(neighbors) * 0.9
        min_range, max_range = val - delta, val + delta

        # Experimental candidates within the range
        candidate_indices = [i for i, (f, a) in enumerate(zip(plot_frequency, plot_amplitude))
                             if min_range <= f <= max_range and a > 5]
        candidate_freqs = [plot_frequency[i] for i in candidate_indices]
        candidate_amps  = [plot_amplitude[i] for i in candidate_indices]

        # Adjust using the most prominent peak if there are enough candidates
        if len(candidate_amps) >= 3:
            peak_indices, props = find_peaks(candidate_amps, prominence=0.02, height=0.1)
            if len(peak_indices) > 0:
                best_idx = peak_indices[np.argmax(props['peak_heights'])]
                final_band[idx] = candidate_freqs[best_idx]
                continue

        # More relaxed adjustment using smooth reflection
        if len(candidate_amps) >= 5:
            delta_ref = min(neighbors) * 0.7
            min_f, max_f = val - delta_ref, val + delta_ref
            indices_window = [i for i, f in enumerate(plot_frequency) if min_f <= f <= max_f]
            freqs_window = [plot_frequency[i] for i in indices_window]
            amps_window  = [plot_amplitude[i] for i in indices_window]

            relaxed_val = detect_smooth_reflection(freqs_window, amps_window, amplitude_min=2, threshold_factor=0.5)
            if relaxed_val is not None:
                final_band[idx] = relaxed_val

    return final_band


def detect_smooth_reflection(frequencies, amplitudes, amplitude_min=0.5, threshold_factor=0.5):
    '''
    Detects a point representing a "smooth relaxation" or zone of symmetry (rounded peak, plateau, symmetric valley).
    Analyzes symmetry of amplitudes around a central point.
    '''
    valid_points = [(f, a) for f, a in zip(frequencies, amplitudes) if a >= amplitude_min]
    if len(valid_points) < 5:
        return None

    frecs, amps = zip(*valid_points)
    central_diffs = []

    for i in range(2, len(amps) - 2):
        x0 = amps[i]
        difs = [amps[i - 2] - x0, amps[i - 1] - x0, x0 - amps[i + 1], x0 - amps[i + 2]]
        avg_diff = np.mean(np.abs(difs))
        central_diffs.append((i, avg_diff))

    if central_diffs:
        best_idx, min_diff = min(central_diffs, key=lambda x: x[1])
        return frecs[best_idx]

    return None


################# 5th step: choose the best option for chains when
# the number of missing peaks is odd #################

def compare_chains_by_peak_quality(chain1, chain2, option1, option2):
    '''
    This function compares two possible bands (generated when there was an odd number
    of missing values). It selects the band with the highest peak intensity in the
    extrapolated positions, keeping the option with better spectral definition.
    '''
    def calculate_peak_intensity(chain):
        total_intensity = 0
        for i in range(1, len(chain) - 1):
            center = chain[i]
            if center is None:
                continue
            left = chain[i - 1]
            right = chain[i + 1]
            if left is not None and right is not None:
                # Considered a peak if center is clearly higher than neighbors
                if center > left and center > right:
                    peak_height = center - max(left, right)
                    total_intensity += peak_height
        return total_intensity

    intensity1 = calculate_peak_intensity(chain1)
    intensity2 = calculate_peak_intensity(chain2)

    return chain1 if intensity1 >= intensity2 else chain2


# Classify frequencies into zones according to boundaries
bands = []
for i in range(num_zones):
    k0, k1 = boundaries[i], boundaries[i+1]
    band = [f for k, f in zip(wave_numbers, peak_frequencies) if k0 <= k <= k1]
    bands.append(band)
    if band:
        print(f"Band {i+1}: min={min(band):.2f}Hz, max={max(band):.2f}Hz, range={max(band)-min(band):.2f}Hz")
print("Bands in final step:", bands)


# Analyze and complete bands depending on whether the missing values are even or odd
bands_possible = analyze_and_complete_bands(bands, num_cylinders)

# Apply extrapolation and store results
new_bands = []
full_bands = []
interpolated_bands = []

for i, possibilities in enumerate(bands_possible):
    best_interpolated = None

    if len(possibilities) == 1:  # Only one possibility
        data = extrapolate_bands(possibilities, num_cylinders)[0]
        best_interpolated = data[2]  # Extract interpolated band
        print("Data with only 1 possibility", data)
    else:
        # Two possibilities: extrapolate both and choose the best
        data1 = extrapolate_bands([possibilities[0]], num_cylinders)[0]
        data2 = extrapolate_bands([possibilities[1]], num_cylinders)[0]
        best_interpolated = compare_chains_by_peak_quality(data1[2], data2[2], [possibilities[0]], [possibilities[1]])
        print("Best interpolated", best_interpolated)
        data = extrapolate_bands([best_interpolated], num_cylinders)[0]
        print("Data after choosing best of 2 options", data)

    new_vals, full_band, band_interp = data
    new_bands.append(new_vals)
    full_bands.append(full_band)
    interpolated_bands.append(band_interp)

print("new_bands 1", new_bands)
new_bands = new_bands1  # Update variable if previously defined
print("new_bands 1", new_bands)


# Adjust final extrapolated values from interpolated bands
final_bands = []
for i, band_interp in enumerate(interpolated_bands):
    new_vals = new_bands[i]
    print("New extrapolated points", new_vals)
    print(f"Band {i+1} interpolated:", band_interp)
    print(f"Band {i+1} new values:", new_vals)
    corrected_band = adjust_extrapolated_values(band_interp, new_vals)
    final_bands.append(corrected_band)
    print(f"Band {i+1} corrected:", corrected_band)


# Collect all new extrapolated peaks that were not in the original bands
new_peaks = []
for original_band, corrected_band in zip(bands, final_bands):
    for val in corrected_band:
        if val not in original_band and val is not None:
            new_peaks.append(val)

print("Original bands:", bands)
print("Corrected bands:", final_bands)
print("New extracted peaks:", new_peaks)


plt.figure(figsize=(10, 6))

# Draw selected peaks
plt.scatter(peak_frequencies, peak_amplitudes, color='magenta', label='Selected peaks', marker='o')

# Labels for peaks
peak_labels = []

# Draw extrapolated peaks with star marker and add text labels
for i, f_peak in enumerate(new_peaks):
    # Buscar el índice más cercano en los datos experimentales
    closest_indices = [j for j, f in enumerate(plot_frequency) if abs(f - f_peak) < 2.49999]
    if closest_indices:
        amp_peak = max(plot_amplitude[j] for j in closest_indices)
        plt.plot(f_peak, amp_peak, '*', color='deeppink', markersize=12)
        label = f'{chr(97 + i)}) {f_peak:.2f} Hz'
        peak_labels.append(label)
        plt.text(f_peak, amp_peak, chr(97 + i), fontsize=10, ha='left', va='bottom', color='black')

# Plot the experimental data curve
plt.plot(plot_frequency, plot_amplitude, label='Data', color='mediumvioletred')

# Custom legend
handles, labels = plt.gca().get_legend_handles_labels()
custom_lines = [Line2D([0], [0], marker='*', color='w', markerfacecolor='violet', markersize=10, label=label) 
                for label in peak_labels]

plt.legend(handles + custom_lines, labels + peak_labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title(f'Peak extraction via extrapolation fitting: {case_name}')
plt.grid(True)
plt.tight_layout()
plt.show()



############# How values change with improvements ###############

plt.figure(figsize=(10, 6))
plt.title(f"Adjusted extrapolated bands: {case_name}")
plt.xlabel("Wave-number k (1/mm)")
plt.ylabel("Frequency (Hz)")

band_labels_set = set()  # Para evitar etiquetas repetidas en la leyenda

for i in range(len(bands_original)):
    color = colors[i % len(colors)]                   # Elegimos color cíclico para cada banda
    band_orig = bands_original[i]                     # Banda original con posibles None
    band_interp = bands_interpolated[i]              # Banda con interpolación aplicada donde faltaba valor
    band_final = final_bands[i]                       # Banda final ajustada tras comparación/extrapolación

    for j in range(target_length):                    # Recorremos cada posición dentro de la banda
        val_orig = band_orig[j]                       # Valor original en la posición j
        val_interp = band_interp[j]                   # Valor interpolado si el original era None
        val_final = band_final[j]                     # Valor final ajustado

        # Determinar posición izquierda y derecha según paridad del índice de banda
        if i % 2 == 0:
            idx_left = target_length - 1 - j
            idx_right = target_length - 1 + j
        else:
            idx_left = j
            idx_right = 2 * target_length - 2 - j

        # Pintar punto izquierdo si está dentro de los límites
        if 0 <= idx_left < len(wave_numbers_full):
            if val_orig is not None:
                if f"Band {i+1}" not in band_labels_set:
                    plt.plot(wave_numbers_full[idx_left], val_orig, 'o', color=color, label=f"Band {i+1}")
                    band_labels_set.add(f"Band {i+1}")
                else:
                    plt.plot(wave_numbers_full[idx_left], val_orig, 'o', color=color)
            elif val_interp is not None:
                plt.plot(wave_numbers_full[idx_left], val_interp, '*', color='grey', markersize=5)
                plt.plot(wave_numbers_full[idx_left], val_final, '^', color=color, markersize=5)

        # Pintar punto derecho si está dentro de los límites
        if 0 <= idx_right < len(wave_numbers_full):
            if val_orig is not None:
                plt.plot(wave_numbers_full[idx_right], val_orig, 'o', color=color)
            elif val_interp is not None:
                plt.plot(wave_numbers_full[idx_right], val_interp, '*', color='grey', markersize=5)
                plt.plot(wave_numbers_full[idx_right], val_final, '^', color=color, markersize=5)

# Elementos de leyenda
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', linestyle='None', label='Obtained value'),
    Line2D([0], [0], marker='*', color='gray', markersize=10, linestyle='None', label='Approx. interpolated value'),
    Line2D([0], [0], marker='^', color='gray', markersize=10, linestyle='None', label='Interpolated value')
]

# Agregar bandas a la leyenda
for i in range(len(bands_original)):
    color = colors[i % len(colors)]
    legend_elements.append(Line2D([0], [0], marker='o', color=color, linestyle='None', label=f'Band {i+1}'))

# Dibujar leyenda al lado derecho
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()














