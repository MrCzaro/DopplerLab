__all__ = ['audio_clipping_metrics']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #6cc9beb9-8989-43a7-be43-8665c7a4a91f
from pathlib import Path
import numpy as np
from scipy.io import wavfile



def audio_clipping_metrics(wav_path, near_peak_threshold=0.98, clip_fraction_threshold=0.001):
    """
    This function computes a candidate audio clipping metric from a WAV file.

    The output is an additive QC flag only. It does not change Doppler waveform
    extraction or beat logic.
    """
    wav_path = Path(wav_path)

    if not wav_path.exists():
        return {
            "audio_clip_fraction": np.nan,
            "audio_peak": np.nan,
            "audio_possible_clipping_candidate": False,
        }

    sample_rate_hz, audio = wavfile.read(wav_path)

    audio = audio.astype(float)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        return {
            "audio_clip_fraction": np.nan,
            "audio_peak": np.nan,
            "audio_possible_clipping_candidate": False,
        }

    peak = float(np.max(np.abs(audio))) + 1e-9
    clip_fraction = float((np.abs(audio) >= near_peak_threshold * peak).mean())
    touches_int16_rail = peak >= 32767

    candidate_clip = bool(
        clip_fraction > clip_fraction_threshold
        and touches_int16_rail
    )

    return {
        "audio_clip_fraction": round(clip_fraction, 5),
        "audio_peak": int(round(peak)),
        "audio_possible_clipping_candidate": candidate_clip,
    }