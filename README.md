# DopplerLab

DopplerLab is an experimental research project for Doppler ultrasound signal exploration.

The project investigates how AVI video, extracted audio, manual calibration, and Mindray native export files can be used to support reproducible Doppler waveform analysis and timing/QC research. The current Python package contains helper functions extracted from the notebook workflow using `nbdev`.

DopplerLab is not a clinical measurement system. Current outputs are research-stage and should be interpreted as experimental, candidate, or metadata-derived unless explicitly validated by project evidence.

## Research Scope

DopplerLab currently focuses on:

- AVI-based Doppler frame reading and image waveform extraction,
- audio extraction and additive audio quality-control checks,
- manual calibration support for AVI-derived Doppler features,
- Mindray native export metadata parsing,
- native PW page/counter structure inspection,
- experimental native `hp_activity` timing/QC sidecar extraction,
- controlled comparison between native metadata/timing signals and AVI/audio-derived evidence.

The project does not claim:

- diagnostic output,
- clinical-grade validation,
- calibrated clinical measurements,
- decoded native spectrogram recovery,
- validated native velocity envelopes,
- native PSV, EDV, RI, PI, or VTI.

## Current Status

The repository contains exploratory notebooks `NB01` through `NB14`, generated validation outputs, and an installable Python package under `dopplerlab/`.

The first package modularization checkpoint was completed from:

```text
NB13 - Native-assisted AVI pipeline improvement validation.ipynb
```

This notebook remains the current nbdev source for the exported helper modules.

## Installation

For runtime/library use:

```bash
pip install .
```

For editable local package development:

```bash
pip install -e .
```

For notebook development, nbdev export work, and exploratory analysis:

```bash
pip install -e ".[dev]"
```

`requirements.txt` is intentionally lightweight and primarily supports local notebook use:

```bash
pip install -r requirements.txt
```

## External Tools

Some exploratory notebooks use `ffmpeg` and `ffprobe` for AVI audio extraction and media metadata checks.

These are external system tools and are not installed by `pip`.

## Package Modules

| Module | Research role |
|---|---|
| `dopplerlab.paths` | Repository-root and project path helpers. |
| `dopplerlab.video` | AVI frame reading with frame index and timing metadata. |
| `dopplerlab.waveform` | Image Doppler waveform extraction, envelope logic, beat morphology helpers, and candidate detector experiments. |
| `dopplerlab.calibration` | Manual calibration helpers and candidate velocity conversion support. |
| `dopplerlab.native_mapping` | Native-to-AVI mapping using manually paired or linked AVI file-size evidence. |
| `dopplerlab.native_metadata` | `DcmRegionPara.txt` parsing for metadata-derived ROI, baseline, velocity scale, and time scale fields. |
| `dopplerlab.native_qc` | Native PW page/counter helpers and experimental `hp_activity` timing/QC sidecar extraction. |
| `dopplerlab.audio_qc` | Additive WAV-based audio clipping QC metrics. |

## Minimal Import Check

```bash
python -c "import dopplerlab; import dopplerlab.native_qc; import dopplerlab.video; print(dopplerlab.__version__)"
```

Expected output includes the current package version:

```text
0.0.1
```

## Example Native Timing/QC Workflow

```python
from pathlib import Path

from dopplerlab.native_metadata import parse_dcm_region_pw
from dopplerlab.native_qc import (
    load_native_pw_pages,
    summarize_native_counter,
    summarize_native_hp_activity_sidecar,
)

native_dir = Path(
    "ultrasound_recordings/"
    "batch_2026_07_09_native/"
    "202607090207400001SMP/"
    "native"
)

pw_path = native_dir / "PW_CinePartition0.bin"

metadata = parse_dcm_region_pw(native_dir)
page_info = load_native_pw_pages(pw_path)
counter_summary = summarize_native_counter(page_info["pages"])
sidecar_summary = summarize_native_hp_activity_sidecar(pw_path)

print(metadata)
print(counter_summary)
print(sidecar_summary)
```

The `hp_activity` sidecar is an experimental timing/QC signal. It is not a decoded native spectrogram, native velocity envelope, or clinical measurement.

## Validation Snapshot

The current modularization checkpoint is supported by evidence from NB13 and NB14:

- NB13 reproduced NB04 V2 full-frame baseline velocity outputs.
- Baseline reproduction evaluated 406 frame rows.
- Baseline median absolute velocity delta was effectively zero.
- Native-to-AVI mapping matched 10/10 native recordings in the original validation batch.
- Fresh July 2026 native/AVI recordings passed exported package smoke tests across 9/9 recordings.
- Fresh native PW files divided cleanly into 1296-byte pages with stable 20 Hz counter grouping.
- Fresh AVI files were readable at 30 FPS with consistent frame dimensions.
- A temporary wheel build and isolated install smoke test passed.

These checks support package usability and research-stage reproducibility. They do not validate clinical measurements.

## Notebook Provenance

The notebooks document the research path and evidence boundaries.

| Notebook | Role |
|---|---|
| `NB01` | Image-based Doppler waveform extraction. |
| `NB02` | AVI audio extraction and audio feature exploration. |
| `NB03` | Image/audio QC integration. |
| `NB04` / `NB04 V2` | AVI-derived feature extraction and batch frame-level reference outputs. |
| `NB05` | Batch screening workflow. |
| `NB06` / `NB06 V2` | Mindray native export exploration and native/audio/image integration checks. |
| `NB07` | Metadata-assisted Doppler feature demonstration. |
| `NB08` | FeParam metadata extraction. |
| `NB09` | `BC_CinePartition1.bin` static raster-like blob exploration. |
| `NB10` | `PW_CinePartition0.bin` native image/spectrogram/state exploration. |
| `NB11` | Native PW flow-wave audit and velocity-boundary checkpoint. |
| `NB12` | Native Mindray file closure and utility summary. |
| `NB13` | Native-assisted AVI pipeline improvement validation and current nbdev export source. |
| `NB14` | Fresh native/AVI batch validation before modularization. |

## nbdev Workflow Note

The current package was exported from a single notebook, `NB13`, using module-specific export directives such as:

```python
#| export paths
#| export native_qc
#| export waveform
```

Do not casually re-run `nb-export` onto the existing generated module files. In this repository setup, re-exporting into existing explicit module files can append duplicate code blocks. Future exports should use a controlled reset-and-export procedure.

## Data and Outputs

Raw ultrasound recordings, extracted audio, frame exports, validation CSVs, and generated notebook artifacts are local research data. They are not part of the installable Python package.

The Python package is intended to contain reusable helper logic only.

## Interpretation Boundary

DopplerLab uses cautious evidence-stage language.

Preferred terms include:

- experimental,
- candidate,
- metadata-derived,
- native timing/QC sidecar,
- not validated,
- consistent with,
- requires visual review.

Avoided claims include:

- ground truth native signal,
- clinical-grade validation,
- diagnostic output,
- decoded native spectrogram,
- validated native velocity envelope,
- native PSV/EDV/RI/PI/VTI,
- calibrated clinical measurement.

## License

DopplerLab is licensed under the Apache License 2.0. See `LICENSE`.

