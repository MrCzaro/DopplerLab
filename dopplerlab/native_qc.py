__all__ = ['PAGE_SIZE_BYTES', 'PAGE_RATE_HZ', 'COUNTER_BYTE_OFFSET', 'HP_ENV_RATE_HZ', 'load_native_pw_pages', 'summarize_native_counter', 'load_native_pw_record_field_b', 'compute_hp_activity', 'compute_activity_envelope', 'estimate_periodic_hr_bpm', 'summarize_native_hp_activity_sidecar']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #f0f47d0f-e6f6-4dea-92f5-707f3cd85595
import numpy as np
import pandas as pd
from pathlib import Path


PAGE_SIZE_BYTES = 1296
PAGE_RATE_HZ = 500.0
COUNTER_BYTE_OFFSET = 1248


def load_native_pw_pages(pw_path, page_size_bytes=PAGE_SIZE_BYTES):
    """
    This function reads PW_CinePartition0.bin as fixed-size native pages.

    Any trailing incomplete bytes are ignored. The returned pages are raw uint8
    bytes and are not interpreted as a decoded spectrogram or velocity signal.
    """
    pw_path = Path(pw_path)

    if not pw_path.exists():
        raise FileNotFoundError(f"Native PW file was not found: {pw_path}")

    raw_bytes = np.fromfile(pw_path, dtype=np.uint8)
    n_pages = raw_bytes.size // page_size_bytes
    trailing_bytes = raw_bytes.size - (n_pages * page_size_bytes)

    if n_pages == 0:
        raise ValueError(f"No complete native PW pages found in {pw_path}")

    pages = raw_bytes[: n_pages * page_size_bytes].reshape(n_pages, page_size_bytes)

    return {
        "pages": pages,
        "n_pages": int(n_pages),
        "file_size_bytes": int(raw_bytes.size),
        "trailing_bytes": int(trailing_bytes),
    }


def summarize_native_counter(
    pages,
    counter_byte_offset=COUNTER_BYTE_OFFSET,
    page_rate_hz=PAGE_RATE_HZ,
):
    """
    This function summarizes the counter-like uint16 field in native PW pages.
    """
    if pages.ndim != 2:
        raise ValueError("pages must be a 2D uint8 array.")

    if pages.shape[1] <= counter_byte_offset + 1:
        raise ValueError("counter byte offset is outside the page width.")

    counter = (
        pages[:, counter_byte_offset].astype(np.uint16)
        | (pages[:, counter_byte_offset + 1].astype(np.uint16) << 8)
    ).astype(np.int64)

    counter_diff = np.diff(counter)
    positive_increment_locations = np.where(counter_diff > 0)[0]

    if positive_increment_locations.size > 2:
        counter_group_pages = float(np.median(np.diff(positive_increment_locations)))
    else:
        counter_group_pages = np.nan

    if np.isfinite(counter_group_pages) and counter_group_pages > 0:
        counter_group_hz = float(page_rate_hz / counter_group_pages)
    else:
        counter_group_hz = np.nan

    return {
        "counter_min": int(counter.min()),
        "counter_max": int(counter.max()),
        "counter_unique": int(np.unique(counter).size),
        "counter_group_pages": counter_group_pages,
        "counter_group_hz": counter_group_hz,
        "counter_resets": int((counter_diff < -100).sum()),
        "counter_monotone_fraction": float((counter_diff >= 0).mean()),
    }

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #0cbd599c-1241-42ad-a3f3-dc5dec3bc2aa
from scipy import signal as sg
from scipy.ndimage import uniform_filter1d


HP_ENV_RATE_HZ = 100.0


def load_native_pw_record_field_b(pw_path):
    """
    This function loads the dynamic record field used for experimental native
    timing/QC sidecar extraction.

    The returned array is an arbitrary-unit compact native page feature. It is
    not interpreted as a decoded spectrogram or velocity envelope.
    """
    page_info = load_native_pw_pages(pw_path)
    pages = page_info["pages"]
    n_pages = page_info["n_pages"]

    records = pages[:, 16:144].reshape(n_pages, 16, 8)

    field_b = (
        np.ascontiguousarray(records[:, :14, 4:8])
        .reshape(-1, 4)
        .view("<f4")
        .reshape(n_pages, 14)
    )

    field_b = np.nan_to_num(field_b.astype(np.float64))

    return field_b, page_info


def compute_hp_activity(field_b, smoothing_window_pages=51):
    """
    This function computes an arbitrary-unit high-pass activity trace from
    compact native PW page records.
    """
    smoothed = uniform_filter1d(
        field_b,
        size=smoothing_window_pages,
        axis=0,
        mode="nearest",
    )

    hp_activity = np.abs(field_b - smoothed).mean(axis=1)

    return hp_activity


def compute_activity_envelope(
    activity,
    page_rate_hz=PAGE_RATE_HZ,
    envelope_rate_hz=HP_ENV_RATE_HZ,
    band_hz=(0.5, 12.0),
):
    """
    This function computes a low-rate envelope from the experimental native
    activity trace for timing/QC estimation.
    """
    activity = np.asarray(activity, dtype=float)

    if activity.ndim != 1:
        raise ValueError("activity must be a 1D array.")

    if not np.isfinite(activity).all():
        raise ValueError("activity contains non-finite values.")

    centered = activity - activity.mean()

    sos = sg.butter(
        N=4,
        Wn=[band_hz[0] / (page_rate_hz / 2.0), band_hz[1] / (page_rate_hz / 2.0)],
        btype="band",
        output="sos",
    )

    filtered = sg.sosfiltfilt(sos, centered)
    envelope = np.abs(sg.hilbert(filtered))

    downsample_factor = int(round(page_rate_hz / envelope_rate_hz))
    envelope_low_rate = envelope[::downsample_factor]

    return envelope_low_rate


def estimate_periodic_hr_bpm(envelope, envelope_rate_hz=HP_ENV_RATE_HZ, low_bpm=40, high_bpm=200):
    """
    This function estimates a dominant HR-like periodicity from an envelope
    autocorrelation.
    """
    envelope = np.asarray(envelope, dtype=float)

    if envelope.ndim != 1:
        raise ValueError("envelope must be a 1D array.")

    signal_centered = envelope - envelope.mean()

    if signal_centered.std() == 0:
        return {
            "autocorr_peak": 0.0,
            "hr_bpm": np.nan,
        }

    n_samples = len(signal_centered)
    fft_values = np.fft.rfft(signal_centered, n=2 * n_samples)
    autocorr = np.fft.irfft(fft_values * np.conj(fft_values))[:n_samples]
    autocorr = autocorr / autocorr[0]

    min_lag = int(envelope_rate_hz * 60.0 / high_bpm)
    max_lag = min(int(envelope_rate_hz * 60.0 / low_bpm), n_samples - 1)

    if max_lag <= min_lag:
        return {
            "autocorr_peak": np.nan,
            "hr_bpm": np.nan,
        }

    local_peak_index = int(np.argmax(autocorr[min_lag:max_lag]))
    lag_samples = min_lag + local_peak_index
    hr_bpm = 60.0 / (lag_samples / envelope_rate_hz)

    return {
        "autocorr_peak": float(autocorr[lag_samples]),
        "hr_bpm": float(hr_bpm),
    }


def summarize_native_hp_activity_sidecar(pw_path):
    """
    This function derives the experimental native hp_activity timing/QC sidecar
    from one raw PW_CinePartition0.bin file.
    """
    field_b, page_info = load_native_pw_record_field_b(pw_path)
    hp_activity = compute_hp_activity(field_b)
    envelope = compute_activity_envelope(hp_activity)
    periodicity = estimate_periodic_hr_bpm(envelope)

    window_length = len(envelope) // 5
    window_hr_values = []

    if window_length > HP_ENV_RATE_HZ * 4:
        for window_index in range(5):
            start = window_index * window_length
            stop = (window_index + 1) * window_length
            window_result = estimate_periodic_hr_bpm(envelope[start:stop])
            window_hr_values.append(window_result["hr_bpm"])

    window_hr_values = np.asarray(window_hr_values, dtype=float)
    finite_window_hr = window_hr_values[np.isfinite(window_hr_values)]

    if len(finite_window_hr) > 2:
        window_hr_iqr_bpm = float(
            np.percentile(finite_window_hr, 75)
            - np.percentile(finite_window_hr, 25)
        )
    else:
        window_hr_iqr_bpm = np.nan

    def subset_hr(column_indices):
        subset_activity = compute_hp_activity(field_b[:, column_indices])
        subset_envelope = compute_activity_envelope(subset_activity)
        return estimate_periodic_hr_bpm(subset_envelope)["hr_bpm"]

    first_half_hr = subset_hr(list(range(0, 7)))
    second_half_hr = subset_hr(list(range(7, 14)))
    even_hr = subset_hr(list(range(0, 14, 2)))
    odd_hr = subset_hr(list(range(1, 14, 2)))

    subset_disagreement_bpm = float(
        np.mean(
            [
                abs(first_half_hr - second_half_hr),
                abs(even_hr - odd_hr),
            ]
        )
    )

    native_specific_reproduced = bool(
        window_hr_iqr_bpm < 8.0
        and subset_disagreement_bpm < 6.0
    )

    return {
        "n_pages": page_info["n_pages"],
        "native_duration_s": page_info["n_pages"] / PAGE_RATE_HZ,
        "hp_activity_finite": bool(np.isfinite(hp_activity).all()),
        "hp_activity_min": float(np.min(hp_activity)),
        "hp_activity_max": float(np.max(hp_activity)),
        "native_hr_bpm": round(periodicity["hr_bpm"], 1),
        "autocorr_peak": round(periodicity["autocorr_peak"], 3),
        "window_hr_iqr_bpm": round(window_hr_iqr_bpm, 1) if np.isfinite(window_hr_iqr_bpm) else np.nan,
        "subset_disagreement_bpm": round(subset_disagreement_bpm, 1),
        "native_specific_reproduced": native_specific_reproduced,
    }