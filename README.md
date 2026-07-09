# DopplerLab

Experimental research helpers for Doppler ultrasound AVI/audio and native-file timing/QC exploration.

This project is in active exploratory development. Current outputs are research-stage and are not clinical measurements, diagnostic outputs, or validated native velocity envelopes.

## Installation

For library/runtime use:

```bash
pip install .
```

For editable local development of the package:

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

Some exploratory notebooks use `ffmpeg` and `ffprobe` for AVI audio extraction and media metadata checks. These are external system tools and are not installed by `pip`.

## Interpretation Boundary

DopplerLab currently provides experimental research helpers. The package does not provide clinical-grade measurements, diagnostic outputs, validated native velocity envelopes, native PSV/EDV/RI/PI/VTI, or calibrated clinical conclusions.

git status --short
Get-Content requirements.txt
