import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, argrelextrema
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
import string
import copy


"""
FIRST PART
In this part we simply load the data, find the most prominent peaks, 
and plot them to study them more thoroughly afterwards. 
This way, we can check if the selected points are correct for the next steps.
"""

# 1.1 Load the data from file into separate lists
data_path = "Caos16cilgrandes_1.dat"

data = np.loadtxt(data_path)
frequency_all = data[:, 0]
amplitude_all = data[:, 1]

# 1.2 Define the parameters that we will use to find missing peaks more easily in later steps
frequency_min_plot = 4500     # Frequency range to analyze and plot
frequency_max_plot = 7800
num_cylinders = 16            # Number of cylinders in the system; gives hints about the number of peaks expected
amplitude_threshold = 0.2     # Minimum amplitude for peaks to be considered
case_description = "16 large cylinders with diaphragms, no order 2"
L = 0.075 * num_cylinders     # Length of the system being measured (used to compute k)
frequency_gap_threshold = 200 # Minimum distance between peaks to consider a new band
k_threshold = 2               
peak_distance = 50            # Minimum distance between peaks in frequency to treat them as distinct
# Define the band shape by marking the missing peaks, so we know where to fill with extrapolated data
bandshape1 = [[None,1,1,1,1,1,1,None,1,None],[1],[None,1,1,None,None]]
bandshape2 = [[None,None,None,None,1,1,1,None,None,None],[1],[None,1,1,None,None]]

# 1.3 Select the data of interest and peaks according to the previously defined parameters
mask_range = (frequency_all >= frequency_min_plot) & (frequency_all <= frequency_max_plot)  
frequency_selected = frequency_all[mask_range]
amplitude_selected = amplitude_all[mask_range]

# Detect peaks in the selected range
peaks, props = find_peaks(amplitude_selected, prominence=0.025)
# Filter peaks by amplitude threshold
filtered_peaks = [i for i in peaks if (amplitude_threshold < amplitude_selected[i])]
frequency_peaks = frequency_selected[filtered_peaks]
amplitude_peaks = amplitude_selected[filtered_peaks]

# 1.4 Estimate the widths of the peaks for later Lorentzian fitting
widths = peak_widths(amplitude_selected, filtered_peaks, rel_height=0.3)[0]
dx = frequency_selected[1] - frequency_selected[0]
fwhm0 = widths * dx

# 1.5 Plot the function to observe the peaks
print(frequency_peaks)
plt.figure(figsize=(10, 6))
plt.plot(frequency_selected, amplitude_selected, label='Experimental Data', color='royalblue')
#plt.scatter(frequency_peaks, amplitude_peaks, color='purple', label='Selected Peaks', zorder=5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title(f'Experimental Data: {case_description}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


"""
SECOND PART
Now we will draw the bands using only the peaks detected by find_peaks,
without adding extrapolated points. This will give us an idea 
of what the shape of the bands will look like.

To do this, we first compute the wavenumbers k and observe where 
the band and sub-band gaps occur. We detect them by calculating the average 
distance between peaks: whenever the gap between two peaks is 
larger than the average, that indicates a band gap. 
The same logic applies for sub-bands, based on the experiment’s nature.
"""

# 2.1 Create a list of wavenumbers (k) and detect the gaps
wavenumber = [i * np.pi / L for i in range(0, len(frequency_peaks))]
k_gaps = []

# Save the band gaps when the frequency jump exceeds the chosen threshold
for i in range(len(frequency_peaks) - 1):
    delta_f = abs(frequency_peaks[i+1] - frequency_peaks[i])

    if delta_f > frequency_gap_threshold:
        k_boundary = (wavenumber[i] + wavenumber[i+1]) / 2
        k_gaps.append(k_boundary)

# Define the zone boundaries
boundaries = [wavenumber[0]] + k_gaps + [wavenumber[-1]]
n_zones = len(boundaries) - 1

# 2.2 Assign each detected peak to its respective band
# Sort the peaks in ascending frequency
frequency_peaks_sorted = sorted(frequency_peaks)

# Find the largest frequency jump (assumed to be the separation between bands)
jumps = np.diff(frequency_peaks_sorted)
largest_jump_index = np.argmax(jumps)
frequency_limit = (frequency_peaks_sorted[largest_jump_index] + frequency_peaks_sorted[largest_jump_index + 1]) / 2

# Separate the peaks into two bands according to the largest gap
band1_peaks = [f for f in frequency_peaks_sorted if f < frequency_limit]
band2_peaks = [f for f in frequency_peaks_sorted if f >= frequency_limit]


# Function to replace 1s in the band structure with detected peak values
def fill_bandshape(bandshape, band_peaks, band_name=""):
    result = []
    ones_total = sum([row.count(1) for row in bandshape])

    if len(band_peaks) < ones_total:
        print(f"Not enough peaks found for {band_name}. Needed {ones_total}, but found {len(band_peaks)}.")
    elif len(band_peaks) > ones_total:
        print(f"Too many peaks found for {band_name}. Needed {ones_total}, but found {len(band_peaks)}.")

    peaks_use = band_peaks[:ones_total]
    it = iter(peaks_use)
    for row in bandshape:
        new_row = []
        for val in row:
            if val == 1:
                try:
                    new_row.append(next(it))
                except StopIteration:
                    new_row.append(None)
            else:
                new_row.append(None)
        result.append(new_row)
    return result


# Apply the function to fill both bands
bandshape1_filled = fill_bandshape(bandshape1, band1_peaks, "Band 1")
bandshape2_filled = fill_bandshape(bandshape2, band2_peaks, "Band 2")

print("Filled band structures:")
print("bandshape1:", bandshape1_filled)
print("bandshape2:", bandshape2_filled)

# 2.3 Make the bands symmetric to check if the results match expectations
# Flatten the lists
intband1 = [element for sublist in bandshape1_filled for element in sublist]
intband2 = [element for sublist in bandshape2_filled for element in sublist]

# Create inverted versions without the first value
intband1_inv = list(reversed(intband1[1:]))
intband2_inv = list(reversed(intband2[1:]))

# Create the complete symmetric versions
bandintcomp1 = intband1_inv + intband1
bandintcomp2 = intband2 + intband2_inv

# 2.4 Plot the results for comparison
kplot = [i * np.pi / L for i in range(-len(intband1)+1, len(intband1))] 
print("kplot:", kplot)

plt.figure(figsize=(10, 6))
plt.plot(kplot, bandintcomp1, 'o', color='royalblue', label='Band 1')
plt.plot(kplot, bandintcomp2, 'o', color='purple', label='Band 2')
plt.xlabel('Wavenumber k (1/m)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Bands formed with peaks detected by find_peaks: {case_description}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


'''
THIRD PART
We fill in the Nones. 

3.1 For this, we must first divide the range where each band takes place.  
As we know, it always follows the form: (big peaks that stand out) - (small peaks hard to see) - (medium but visible peaks).  
So, the central sub-band will be the band range minus the ranges of sub-bands 1 and 3.  

For the other sub-bands, what we do is take one peak, for example, from the first sub-band  
(we know this because they are in lists within lists) and then move until we reach the values  
where the amplitude decreases significantly and reaches a threshold that we call `band_peak_threshold`.  
When the amplitude reaches that threshold, we assume the band range has ended — both on the lower and higher frequency sides.  
The same logic applies to the rest.  

3.2 Once we know the distribution of the regions, we look for the peaks as follows:  
We check where Nones are missing under the following cases:  

1. If the None is at the beginning, then we search for a peak or inflection point in the range from the start of the band until the first already stored peak.  
   If more than one peak exists in that range, we search for them all, but we pick the most prominent one.  
   The same logic applies to inflection points, since sometimes peaks overlap too much.  

2. If the None is between two already detected peaks, then we simply search between the closest detected peaks.  

3. If the band is completely empty, we search across the entire band range.  

...  
With these cases, it should be clear what needs to be done. Important: we do NOT allow the search range to start exactly at the previous peak,  
but rather with a small delta before it, which we call `peaksearch_delta`.  
The key point here is to know the regions where peaks will appear and their order, since this allows us to know where to search for the missing peak.
'''

# 3.1 Define the ranges of each sub-band (and band)
band_peak_threshold = 0.2  # Drop delta to determine sub-band boundaries

def find_subband_limits(amplitude, frequency, central_peak, delta=0.2):
    '''
    Find the range of the sub-band.  
    For this, we take one peak of the sub-band (always if it is sub-band 1 or 3) and  
    then check in which frequencies the amplitude decreases until a given delta (default 0.2).  
    With this, we obtain the limits of sub-bands 1 and 3.  
    Sub-band 2 will simply be the space between them.
    '''
    idx_peak = np.argmin(np.abs(frequency - central_peak))

    # Extend to the left
    idx_left = idx_peak
    while idx_left > 0 and amplitude[idx_left] > delta:
        idx_left -= 1

    # Extend to the right
    idx_right = idx_peak
    while idx_right < len(amplitude) - 1 and amplitude[idx_right] > delta:
        idx_right += 1

    return frequency[idx_left], frequency[idx_right]

# Process bands
subbands_ranges = {}  # Store ranges here

# --- Band 1 ---
# Sub-band 1 (find the first peak)
peak_b1s1 = next((p for row in bandshape1_filled for p in row if p is not None), None)
if peak_b1s1:
    liminf11, limsup11 = find_subband_limits(amplitude_selected, frequency_selected, peak_b1s1, delta=band_peak_threshold)
    subbands_ranges['band1_subband1'] = [liminf11, limsup11]

# Sub-band 3 (find the last peak)
peak_b1s3 = next((p for row in reversed(bandshape1_filled) for p in reversed(row) if p is not None), None)
if peak_b1s3:
    liminf13, limsup13 = find_subband_limits(amplitude_selected, frequency_selected, peak_b1s3, delta=band_peak_threshold)
    subbands_ranges['band1_subband3'] = [liminf13, limsup13]

# Sub-band 2 (middle one)
subbands_ranges['band1_subband2'] = [limsup11, liminf13] if 'band1_subband1' in subbands_ranges and 'band1_subband3' in subbands_ranges else None

# --- Band 2 ---
# Sub-band 1 (first peak of band 2)
peak_b2s1 = next((p for row in bandshape2_filled for p in row if p is not None), None)
if peak_b2s1:
    liminf21, limsup21 = find_subband_limits(amplitude_selected, frequency_selected, peak_b2s1, delta=band_peak_threshold)
    subbands_ranges['band2_subband1'] = [liminf21, limsup21]

# Sub-band 3 (last peak of band 2)
peak_b2s3 = next((p for row in reversed(bandshape2_filled ) for p in reversed(row) if p is not None), None)
if peak_b2s3:
    liminf23, limsup23 = find_subband_limits(amplitude_selected, frequency_selected, peak_b2s3, delta=band_peak_threshold)
    subbands_ranges['band2_subband3'] = [liminf23, limsup23]

# Sub-band 2 (central one)
subbands_ranges['band2_subband2'] = [limsup21, liminf23] if 'band2_subband1' in subbands_ranges and 'band2_subband3' in subbands_ranges else None

# Print results
for key, value in subbands_ranges.items():
    print(f"{key} = {value}")

# Create a list of lists with ranges in the same order as the sub-bands
bands_subbands_ranges = [
    [  # Band 1
        subbands_ranges.get('band1_subband1', None),
        subbands_ranges.get('band1_subband2', None),
        subbands_ranges.get('band1_subband3', None)
    ],
    [  # Band 2
        subbands_ranges.get('band2_subband1', None),
        subbands_ranges.get('band2_subband2', None),
        subbands_ranges.get('band2_subband3', None)
    ]
]


# 3.2 Search for new peaks
def detect_smooth_relaxation(frequencies, amplitudes, min_amplitude=0.5):
    '''
    Detects a smooth zone resembling a peak or inflection point: low variation around a point.  
    '''
    valid_points = [(f, a) for f, a in zip(frequencies, amplitudes) if a >= min_amplitude]  # Filter only points with enough amplitude (discard weak noise).
    if len(valid_points) < 3:  # If too few points, skip.
        return None

    freqs, amps = zip(*valid_points)
    centered_diffs = []

    # Calculate local symmetric variation around each point (average differences).
    for i in range(2, len(amps) - 2):
        x0 = amps[i]
        diffs = [amps[i - 2] - x0, amps[i - 1] - x0, x0 - amps[i + 1], x0 - amps[i + 2]]
        mean_diff = np.mean(np.abs(diffs))
        centered_diffs.append((i, mean_diff))

    # Select the point with the smallest variation (smoothest)
    if centered_diffs:
        idx_best, min_diff = min(centered_diffs, key=lambda x: x[1])
        return freqs[idx_best]

    return None


def fill_missing_subbands(filled_bands, bands_subbands_ranges, all_frequencies, all_amplitudes, min_peak_distance=20, peaksearch_delta=10, repeat_threshold=5):
    '''
    This function fills the None values in spectral bands by searching for nearby peaks (or smooth zones if no peaks are found).
    '''
    def filter_peaks_by_min_distance(freqs, amps, min_dist):
        # Filters peaks so they are at least `min_dist` apart (avoid duplicates).
        if not freqs:
            return [], []
        ordered_peaks = sorted(zip(freqs, amps), key=lambda x: x[1], reverse=True)
        filtered_peaks = []
        for f, a in ordered_peaks:
            if all(abs(f - f_exist) > min_dist for f_exist, _ in filtered_peaks):
                filtered_peaks.append((f, a))
        filtered_peaks.sort(key=lambda x: x[0])
        if not filtered_peaks:
            return [], []
        freqs_final, amps_final = zip(*filtered_peaks)
        return list(freqs_final), list(amps_final)

    def is_too_close(value, lst, threshold):
        # Returns True if value is closer than threshold to any element in lst.
        return any(v is not None and abs(value - v) <= threshold for v in lst)

    completed_bands = []
    # Iterate over each band (filled_bands) with its sub-band ranges
    for idx_band, (subbands, subband_ranges) in enumerate(zip(filled_bands, bands_subbands_ranges)):
        new_band = []
        for idx_subband, (subband, sub_range) in enumerate(zip(subbands, subband_ranges)):
            new_subband = []
            sub = subband.copy()
            for idx_pos, val in enumerate(sub):
                if val is not None:
                    new_subband.append(val)
                    continue

                if sub_range is None:
                    new_subband.append(None)
                    continue

                f_min, f_max = sub_range
                idxs = np.where((all_frequencies >= f_min) & (all_frequencies <= f_max))[0]
                if len(idxs) < 5:
                    new_subband.append(None)
                    continue

                freqs_local = all_frequencies[idxs]
                amps_local = all_amplitudes[idxs]

                # Determine left and right peaks
                peak_left = next((p for p in reversed(sub[:idx_pos]) if p is not None), None)
                peak_right = next((p for p in sub[idx_pos+1:] if p is not None), None)

                f_search_start = f_min
                f_search_end = f_max

                # Adjust search range to avoid overlapping known peaks
                if peak_left and peak_right:
                    f_search_start = peak_left + peaksearch_delta
                    f_search_end = peak_right - peaksearch_delta
                elif peak_left:
                    f_search_start = peak_left + peaksearch_delta
                elif peak_right:
                    f_search_end = peak_right - peaksearch_delta

                idx_range = np.where((all_frequencies >= f_search_start) & (all_frequencies <= f_search_end))[0]
                if len(idx_range) < 5:
                    new_subband.append(None)
                    continue

                freqs_search = all_frequencies[idx_range]
                amps_search = all_amplitudes[idx_range]

                # Use known peaks to set a reasonable minimum amplitude
                known_peaks = [v for v in sub if v is not None]
                if known_peaks:
                    known_amps = [all_amplitudes[np.argmin(np.abs(all_frequencies - v))] for v in known_peaks]
                    min_amp = np.min(known_amps) * 0.25
                else:
                    min_amp = 0.05

                # Find new peaks
                pks, _ = find_peaks(amps_search, prominence=0.005)
                pks = [i for i in pks if amps_search[i] >= min_amp]
                cand_freqs = [freqs_search[i] for i in pks]
                cand_amps = [amps_search[i] for i in pks]

                cand_freqs, _ = filter_peaks_by_min_distance(cand_freqs, cand_amps, min_peak_distance)

                already_used = [v for v in sub if v is not None] + new_subband
                chosen = next((f for f in cand_freqs if not is_too_close(f, already_used, repeat_threshold)), None)

                if chosen is not None:
                    new_subband.append(chosen)
                    print(f"Sub-band [{idx_band}, {idx_subband}, pos {idx_pos}] → Peak at {chosen:.2f} Hz")
                    continue

                # If no peak, try inflection point
                inflection = detect_smooth_relaxation(freqs_search, amps_search, min_amplitude=min_amp)
                if inflection and not is_too_close(inflection, already_used, repeat_threshold):
                    new_subband.append(inflection)
                    print(f"Sub-band [{idx_band}, {idx_subband}, pos {idx_pos}] → Inflection at {inflection:.2f} Hz")
                else:
                    new_subband.append(None)

            new_band.append(new_subband)

        completed_bands.append(new_band)

    return completed_bands


bands_input = [bandshape1_filled, bandshape2_filled]
completed_bands = fill_missing_subbands(bands_input, bands_subbands_ranges, frequency_selected,
    amplitude_selected, min_peak_distance=peak_distance)

# Finally, make sure sub-bands are ordered since sometimes they are not
def sort_subbands(band):
    band_sorted = []
    for subband in band:
        sorted_vals = sorted([x for x in subband if x is not None])
        n_nones = subband.count(None)
        subband_sorted = sorted_vals + [None] * n_nones
        band_sorted.append(subband_sorted)
    return band_sorted



# Sort after assignment
final_band1 = sort_subbands(completed_bands[0])
final_band2 = sort_subbands(completed_bands[1])
print("final_band1 sorted:", final_band1)
print("final_band2 sorted:", final_band2)

print("final_band1:", final_band1)
print("final_band2:", final_band2)


# 1. Flatten the final band data
new_frequencies = []
for band in completed_bands:
    for subband in band:
        for f in subband:
            if f is not None:
                new_frequencies.append(f)

# 2. Get the corresponding amplitudes for each new frequency
new_amplitudes = [amplitude_selected[np.argmin(np.abs(frequency_selected - f))] for f in new_frequencies]

# 3. Plot the data
plt.figure(figsize=(10, 6))
plt.plot(frequency_selected, amplitude_selected, label='Data', color='royalblue')

# Initial points found by find_peaks
plt.scatter(frequency_peaks, amplitude_peaks, color='purple', label='Selected peaks (original)', zorder=5)

# New points added (automatically detected)
plt.scatter(new_frequencies, new_amplitudes, color='orange', label='Added peaks/inflections', zorder=6, marker='x', s=80)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title(f'Detected peaks: {case_description}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


'''
FOURTH PART
Graph of topological peaks. As always, the topological peaks are formed
in the middle of the band, which means they will always belong to the second subband.
So we only need to take those points, put them into a list and mark them with stars.
I was also going to mark band and subband jumps, but the way I defined the
bands they do not occur at the same k, and adding them could be confusing.
'''

# Create the full band 1
band1_linear = [f for row in final_band1 for f in row]
band1_linear_inv = list(reversed(band1_linear[1:]))
band1_full = band1_linear_inv + band1_linear

# Create the full band 2
band2_linear = [f for row in final_band2 for f in row]
band2_linear_inv = list(reversed(band2_linear[1:]))
band2_full = band2_linear + band2_linear_inv

# Define the k values for both full bands
k_band1 = [i * np.pi / L for i in range(-len(band1_linear)+1, len(band1_linear))]
k_band2 = [i * np.pi / L for i in range(-len(band2_linear)+1, len(band2_linear))]

# Get the topological values (subband2 = index 1)
topo1 = [f for f in final_band1[1] if f is not None]
topo2 = [f for f in final_band2[1] if f is not None]

# Now we search for the topological points.

# Band 1
k_topo1 = []
f_topo1 = []
for f in topo1:
    indices = [i for i, val in enumerate(band1_full) if val is not None and np.isclose(val, f, atol=1e-6)]
    for idx in indices:
        k_topo1.append(k_band1[idx])
        f_topo1.append(f)

# Band 2
k_topo2 = []
f_topo2 = []
for f in topo2:
    indices = [i for i, val in enumerate(band2_full) if val is not None and np.isclose(val, f, atol=1e-6)]
    for idx in indices:
        k_topo2.append(k_band2[idx])
        f_topo2.append(f)


plt.figure(figsize=(10, 6))
plt.plot(k_band1, band1_full, 'o', color='royalblue', label='Band 1')
plt.plot(k_band2, band2_full, 'o', color='purple', label='Band 2')
plt.plot(k_topo1, f_topo1, 'r*', markersize=12, label='Topological points')
plt.plot(k_topo2, f_topo2, 'r*', markersize=12)
plt.xlabel('Wave number k (1/m)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Bands: {case_description}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


'''
FIFTH PART
Approximation graph. In this section we apply the SSH function
to the first band and use it to obtain the approximation of what we should
have found in the second band if we had detected all points. 
We simply define the function to be used, then apply curve_fit to obtain the fitting parameters
for both the first and the second band.
'''

# Define the SSH function
def SSH_dispersion(k, v, w):
    return np.sqrt(v**2 + w**2 + 2*v*w*np.cos(k))

# Convert lists to arrays with np.nan where there are None values
def to_array_with_nan(lst):
    return np.array([float(x) if x is not None else np.nan for x in lst])

# Flatten lists with np.ravel to avoid nested structures (required for curve_fit)
k_band1 = np.ravel(np.array(k_band1, dtype=float)) 
k_band2 = np.ravel(np.array(k_band2, dtype=float))
band1 = np.ravel(to_array_with_nan(band1_full))
band2 = np.ravel(to_array_with_nan(band2_full))

# Match lengths without removing data (needed because of Nones)
min_len1 = min(len(k_band1), len(band1))
k_band1 = k_band1[:min_len1]
band1 = band1[:min_len1]

min_len2 = min(len(k_band2), len(band2))
k_band2 = k_band2[:min_len2]
band2 = band2[:min_len2]

# Fit only with positive k values (otherwise symmetry gives a straight line)
k_max = np.pi * num_cylinders / L
mask_positive = k_band1 >= 0
k_positive = k_band1[mask_positive]
band1_positive = band1[mask_positive]

# Scale k to [-pi, pi]
k_scaled_fit = k_positive * (np.pi / k_max)

# Filter NaN or Inf only in fitting
mask_finite = np.isfinite(k_scaled_fit) & np.isfinite(band1_positive)
k_fit = k_scaled_fit[mask_finite]
band_fit = band1_positive[mask_finite]

# Fit data to obtain the approximation 
params, _ = curve_fit(SSH_dispersion, k_fit, band_fit, p0=[1.0, 1.0])
v_fit, w_fit = params
print(f"Fitted parameters (initial fit):\n v = {v_fit:.4f}, w = {w_fit:.4f}")

# Redefine k to full range and compute approximations for both bands
k_full = np.linspace(-k_max, k_max, 500)
k_scaled_full = k_full * (np.pi / k_max)
band1_ssh = SSH_dispersion(k_scaled_full, v_fit, w_fit)
band2_ssh = SSH_dispersion(k_scaled_full[::-1], v_fit, w_fit) + (band2[3] - band1[3])

# k for full curve without point mismatch
k1 = np.linspace(-k_max, k_max, len(band1))
k2 = np.linspace(-k_max, k_max, len(band2))

# Final fit with clean band1 (without altering the original)
mask_finite_final = np.isfinite(k1) & np.isfinite(band1)
k1_clean = k1[mask_finite_final]
band1_clean = band1[mask_finite_final]

params, _ = curve_fit(SSH_dispersion, k1_clean, band1_clean, p0=[1.0, 1.0])
v_fit, w_fit = params
print(f"Fitted parameters (full fit):\n v = {v_fit:.4f}, w = {w_fit:.4f}")

# Evaluate with all points (keeping original data)
band1_theo = SSH_dispersion(k1, v_fit, w_fit)
band2_theoretical = SSH_dispersion(k2, v_fit, w_fit) + (band2[3] - band1[3])

midpoint_b2 = np.nanmean(band2)
mean_band1_theo = np.mean(band1_ssh)
band2_theo = midpoint_b2 - (band1_ssh - mean_band1_theo)

# Plot to check results
plt.figure(figsize=(10,6))

# Experimental band 1
plt.plot(k_band1, band1, 'o', color='royalblue', label='Band 1')

# Experimental band 2
plt.plot(k_band2, band2, 'o', color='purple', label='Band 2')

# Topological points
plt.plot(k_topo1, f_topo1, 'r*', markersize=12, label='Topological points')
plt.plot(k_topo2, f_topo2, 'r*', markersize=12)

# Theoretical band 1
plt.plot(k_full, band1_ssh, '-', color='red', label='Theoretical band 1 (SSH fit)')

# Theoretical band 2
plt.plot(k_full, band2_theo, '-', color='green', label='Theoretical band 2 (from band 1)')

plt.xlabel('Wave number k (1/m)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Bands using SSH approximation: {case_description}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
















