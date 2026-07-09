__all__ = ['extract_envelope_directional', 'extract_doppler_waveform', 'analyze_beats', 'summarize_complete_beats', 'autocorr_guided_peaks', 'analyze_waveform_with_replaced_peaks']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #e3643131-bf3c-429a-87d0-06f9aafa2eb1
import numpy as np
import pandas as pd
import cv2
from scipy.signal import medfilt, savgol_filter, find_peaks


def extract_envelope_directional(mask, baseline, direction="auto", min_segment_length=5):
    """
    This function extracts the Doppler spectrum envelope from a binary mask.
    """
    if direction not in ["auto", "above", "below"]:
        raise ValueError("direction must be 'auto', 'above', or 'below'")
    if baseline <= 0 or baseline >= mask.shape[0] - 1:
        raise ValueError("Baseline is outside valid ROI range.")
    if min_segment_length < 1:
        raise ValueError("min_segment_length must be >= 1.")

    above_area = int(mask[:baseline, :].sum())
    below_area = int(mask[baseline + 1:, :].sum())

    if direction == "auto":
        selected_direction = "above" if above_area >= below_area else "below"
    else:
        selected_direction = direction

    envelope = np.full(mask.shape[1], np.nan)

    for x in range(mask.shape[1]):
        if selected_direction == "above":
            ys = np.where(mask[:baseline, x])[0]
        else:
            ys_local = np.where(mask[baseline + 1:, x])[0]
            ys = baseline + 1 + ys_local

        if len(ys) == 0:
            continue

        breaks = np.where(np.diff(ys) > 1)[0] + 1
        segments = np.split(ys, breaks)
        valid_segments = [segment for segment in segments if len(segment) >= min_segment_length]

        if len(valid_segments) == 0:
            continue

        if selected_direction == "above":
            envelope[x] = min(segment.min() for segment in valid_segments)
        else:
            envelope[x] = max(segment.max() for segment in valid_segments)

    if selected_direction == "above":
        velocity = baseline - envelope
    else:
        velocity = envelope - baseline

    return envelope, velocity, selected_direction


def extract_doppler_waveform(
    frame_rgb,
    x_min=80,
    x_max=640,
    y_min=230,
    y_max=510,
    threshold=80,
    direction="auto",
    min_segment_length=5,
    trim_left=10,
    trim_right=10,
    median_kernel_size=5,
    savgol_window_length=21,
    savgol_polyorder=3,
    peak_height=80,
    peak_distance=100,
):
    """
    This function extracts and smooths a Doppler waveform from a single RGB video frame.
    """
    roi = frame_rgb[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    row_sums = mask.sum(axis=1)
    baseline = int(np.argmax(row_sums))

    envelope, velocity_raw, selected_direction = extract_envelope_directional(
        mask=mask,
        baseline=baseline,
        direction=direction,
        min_segment_length=min_segment_length,
    )

    if trim_left < 0 or trim_right < 0:
        raise ValueError("trim_left and trim_right must be non-negative.")
    if trim_left + trim_right >= len(envelope):
        raise ValueError("trim_left + trim_right is too large for signal length.")

    keep_slice = slice(trim_left, None) if trim_right == 0 else slice(trim_left, -trim_right)

    envelope = envelope[keep_slice]
    velocity_raw = velocity_raw[keep_slice]
    x_offset = trim_left
    valid = np.where(~np.isnan(envelope))[0]

    if len(valid) == 0:
        raise ValueError("No valid envelope points found. Check ROI, threshold, Doppler direction, min_segment_length, or trim settings.")

    envelope_interp = np.interp(np.arange(len(envelope)), valid, envelope[valid])
    velocity_raw = np.interp(np.arange(len(velocity_raw)), valid, velocity_raw[valid])

    if median_kernel_size % 2 == 0:
        median_kernel_size += 1
    velocity_med = medfilt(velocity_raw, kernel_size=median_kernel_size)

    if savgol_window_length % 2 == 0:
        savgol_window_length += 1
    if savgol_window_length >= len(velocity_med):
        savgol_window_length = len(velocity_med) - 1
        if savgol_window_length % 2 == 0:
            savgol_window_length -= 1
    if savgol_window_length <= savgol_polyorder:
        raise ValueError("savgol_window_length must be greater than savgol_polyorder.")

    velocity_smooth = savgol_filter(
        velocity_med,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
    )

    peaks_local, peak_props = find_peaks(
        velocity_smooth,
        height=peak_height,
        distance=peak_distance,
    )
    peaks_global = peaks_local + x_offset

    return {
        "roi": roi,
        "gray": gray,
        "mask": mask,
        "baseline": baseline,
        "direction": selected_direction,
        "min_segment_length": min_segment_length,
        "trim_left": trim_left,
        "trim_right": trim_right,
        "x_offset": x_offset,
        "envelope": envelope_interp,
        "velocity_raw": velocity_raw,
        "velocity_med": velocity_med,
        "velocity_smooth": velocity_smooth,
        "peaks_local": peaks_local,
        "peaks_global": peaks_global,
        "peaks": peaks_local,
        "peak_props": peak_props,
    }

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #c3dba076-5943-46a2-92c8-cc0b3b8857b0
def analyze_beats(
    result,
    min_valley_distance_from_psv=10,
    secondary_peak_search_start_after_valley=5,
    secondary_peak_search_end_after_psv=70,
    min_time_to_valley_px=15,
    max_time_to_valley_fraction=0.65,
    require_secondary_peak=True,
):
    """
    This function extracts beat-level morphology features from a Doppler waveform.
    """
    velocity = result["velocity_smooth"]
    peaks_local = result["peaks_local"]
    peaks_global = result["peaks_global"]
    x_offset = result["x_offset"]

    rows = []

    for i in range(len(peaks_local) - 1):
        psv_local = int(peaks_local[i])
        next_psv_local = int(peaks_local[i + 1])
        psv_global = int(peaks_global[i])
        next_psv_global = int(peaks_global[i + 1])

        beat_start = psv_local
        beat_end = next_psv_local

        if beat_end <= beat_start:
            continue

        cycle_length_px = int(next_psv_local - psv_local)
        psv_value = float(velocity[psv_local])

        valley_search_start = beat_start + min_valley_distance_from_psv
        valley_search_end = beat_end

        if valley_search_start >= valley_search_end:
            continue

        valley_region = velocity[valley_search_start:valley_search_end]

        if len(valley_region) == 0:
            continue

        first_valley_local = int(valley_search_start + np.argmin(valley_region))
        first_valley_global = int(first_valley_local + x_offset)
        first_valley_value = float(velocity[first_valley_local])
        time_to_valley_px = int(first_valley_local - psv_local)

        secondary_start = first_valley_local + secondary_peak_search_start_after_valley
        secondary_end = min(beat_start + secondary_peak_search_end_after_psv, beat_end)

        if secondary_start < secondary_end:
            secondary_region = velocity[secondary_start:secondary_end]
            if len(secondary_region) > 0:
                secondary_peak_local = int(secondary_start + np.argmax(secondary_region))
                secondary_peak_global = int(secondary_peak_local + x_offset)
                secondary_peak_value = float(velocity[secondary_peak_local])
            else:
                secondary_peak_local = np.nan
                secondary_peak_global = np.nan
                secondary_peak_value = np.nan
        else:
            secondary_peak_local = np.nan
            secondary_peak_global = np.nan
            secondary_peak_value = np.nan

        time_to_secondary_peak_px = (
            int(secondary_peak_local - psv_local)
            if not np.isnan(secondary_peak_local)
            else np.nan
        )

        edv_proxy = first_valley_value
        ri_proxy = (psv_value - edv_proxy) / psv_value if psv_value > 0 else np.nan
        valley_to_psv_ratio = first_valley_value / psv_value if psv_value > 0 else np.nan
        secondary_to_psv_ratio = (
            secondary_peak_value / psv_value
            if psv_value > 0 and not np.isnan(secondary_peak_value)
            else np.nan
        )

        quality_reasons = []

        if time_to_valley_px < min_time_to_valley_px:
            quality_reasons.append("valley_too_early")
        if time_to_valley_px > cycle_length_px * max_time_to_valley_fraction:
            quality_reasons.append("valley_too_late")
        if require_secondary_peak and np.isnan(secondary_peak_local):
            quality_reasons.append("missing_secondary_peak")

        is_complete_beat = len(quality_reasons) == 0
        beat_quality_reason = "ok" if is_complete_beat else ";".join(quality_reasons)

        rows.append(
            {
                "beat_id": i + 1,
                "is_complete_beat": is_complete_beat,
                "beat_quality_reason": beat_quality_reason,
                "psv_local": psv_local,
                "psv_global": psv_global,
                "next_psv_local": next_psv_local,
                "next_psv_global": next_psv_global,
                "psv_value_px": psv_value,
                "first_valley_local": first_valley_local,
                "first_valley_global": first_valley_global,
                "first_valley_value_px": first_valley_value,
                "secondary_peak_local": secondary_peak_local,
                "secondary_peak_global": secondary_peak_global,
                "secondary_peak_value_px": secondary_peak_value,
                "cycle_length_px": cycle_length_px,
                "time_to_valley_px": time_to_valley_px,
                "time_to_secondary_peak_px": time_to_secondary_peak_px,
                "valley_to_psv_ratio": valley_to_psv_ratio,
                "secondary_to_psv_ratio": secondary_to_psv_ratio,
                "ri_proxy": ri_proxy,
            }
        )

    return pd.DataFrame(rows)


def summarize_complete_beats(beat_df):
    """
    This function aggregates complete Doppler beats into recording-level features.
    """
    if len(beat_df) == 0:
        raise ValueError("beat_df is empty.")

    complete_beats = beat_df[beat_df["is_complete_beat"]].copy()
    n_total = len(beat_df)
    n_complete = len(complete_beats)

    if n_complete == 0:
        return pd.DataFrame(
            [{"n_total_beats": n_total, "n_complete_beats": 0, "complete_fraction": 0.0}]
        )

    summary = {
        "n_total_beats": n_total,
        "n_complete_beats": n_complete,
        "complete_fraction": n_complete / n_total,
        "mean_psv_px": complete_beats["psv_value_px"].mean(),
        "std_psv_px": complete_beats["psv_value_px"].std(),
        "mean_edv_proxy_px": complete_beats["first_valley_value_px"].mean(),
        "mean_ri_proxy": complete_beats["ri_proxy"].mean(),
        "std_ri_proxy": complete_beats["ri_proxy"].std(),
        "mean_cycle_length_px": complete_beats["cycle_length_px"].mean(),
        "mean_time_to_valley_px": complete_beats["time_to_valley_px"].mean(),
        "mean_time_to_secondary_peak_px": complete_beats["time_to_secondary_peak_px"].mean(),
        "mean_secondary_to_psv_ratio": complete_beats["secondary_to_psv_ratio"].mean(),
    }

    return pd.DataFrame([summary])

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #ec215bcf-1424-4984-8ced-644d66e92c15
from scipy.signal import find_peaks


def autocorr_guided_peaks(velocity_smooth, min_cycle_length_px=60, max_cycle_length_px=400):
    """
    This function proposes candidate peaks using an autocorrelation-derived
    cycle-length prior.

    It is an experimental beat detector candidate, not the default detector.
    """
    velocity_smooth = np.asarray(velocity_smooth, dtype=float)

    if velocity_smooth.ndim != 1:
        raise ValueError("velocity_smooth must be a 1D array.")

    centered = velocity_smooth - np.mean(velocity_smooth)

    if centered.std() == 0:
        return np.array([], dtype=int)

    n_samples = len(centered)

    if n_samples <= min_cycle_length_px + 5:
        fallback_peaks, _ = find_peaks(
            velocity_smooth,
            height=80,
            distance=100,
        )
        return fallback_peaks

    fft_values = np.fft.rfft(centered, n=2 * n_samples)
    autocorr = np.fft.irfft(fft_values * np.conj(fft_values))[:n_samples]
    autocorr = autocorr / autocorr[0]

    search_stop = min(n_samples - 1, max_cycle_length_px)

    if search_stop <= min_cycle_length_px:
        fallback_peaks, _ = find_peaks(
            velocity_smooth,
            height=80,
            distance=100,
        )
        return fallback_peaks

    cycle_length_px = min_cycle_length_px + int(
        np.argmax(autocorr[min_cycle_length_px:search_stop])
    )

    candidate_distance = max(1, int(0.7 * cycle_length_px))

    candidate_peaks, _ = find_peaks(
        velocity_smooth,
        height=80,
        distance=candidate_distance,
    )

    return candidate_peaks


def analyze_waveform_with_replaced_peaks(waveform_result, peaks_local):
    """
    This function recomputes beat morphology after replacing the detected peaks.
    """
    replaced = dict(waveform_result)
    replaced["peaks_local"] = np.asarray(peaks_local, dtype=int)
    replaced["peaks_global"] = np.asarray(peaks_local, dtype=int) + int(waveform_result["x_offset"])
    replaced["peaks"] = np.asarray(peaks_local, dtype=int)

    if len(replaced["peaks_local"]) < 2:
        return pd.DataFrame()

    return analyze_beats(replaced)