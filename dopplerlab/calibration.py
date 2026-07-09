__all__ = ['fit_velocity_calibration_from_points', 'derive_max_velocity_cm_s_from_waveform']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #0884fc6b-81d3-4b5c-8890-c5199c520f77
import numpy as np
import pandas as pd


def fit_velocity_calibration_from_points(calibration_points_df):
    """
    This function fits Doppler velocity calibration from manually selected scale points.
    """
    required_columns = ["calibration_group", "velocity_cm_s", "y_global_px"]
    missing_columns = [
        column for column in required_columns
        if column not in calibration_points_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Calibration points table is missing required columns:\n"
            + "\n".join(missing_columns)
        )

    calibration_rows = []

    for calibration_group, group_df in calibration_points_df.groupby("calibration_group"):
        group_df = group_df.copy()

        if len(group_df) < 2:
            raise ValueError(f"Calibration group '{calibration_group}' has fewer than 2 points.")

        y = group_df["y_global_px"].to_numpy(dtype=float)
        v = group_df["velocity_cm_s"].to_numpy(dtype=float)

        slope_cm_s_per_px, intercept_cm_s = np.polyfit(y, v, deg=1)
        predicted_velocity_cm_s = slope_cm_s_per_px * y + intercept_cm_s
        residuals_cm_s = v - predicted_velocity_cm_s
        cm_s_per_px = abs(float(slope_cm_s_per_px))

        baseline_candidates = group_df[
            np.isclose(group_df["velocity_cm_s"].astype(float), 0.0)
        ]

        if len(baseline_candidates) > 0:
            baseline_y_global_px = float(baseline_candidates["y_global_px"].iloc[0])
        else:
            baseline_y_global_px = float(-intercept_cm_s / slope_cm_s_per_px)

        calibration_rows.append(
            {
                "calibration_group": calibration_group,
                "n_points": len(group_df),
                "slope_cm_s_per_px": float(slope_cm_s_per_px),
                "intercept_cm_s": float(intercept_cm_s),
                "cm_s_per_px": cm_s_per_px,
                "baseline_y_global_px": baseline_y_global_px,
                "baseline_y_roi_px": baseline_y_global_px - 230.0,
                "max_abs_residual_cm_s": float(np.max(np.abs(residuals_cm_s))),
                "mean_abs_residual_cm_s": float(np.mean(np.abs(residuals_cm_s))),
            }
        )

    return pd.DataFrame(calibration_rows)

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #4af36a7d-a68a-4099-9cc5-6947e607ff91
import numpy as np


def derive_max_velocity_cm_s_from_waveform(
    waveform_result,
    cm_s_per_px,
    roi_y_min=230,
    native_baseline_global=None,
):
    """
    This function derives a max-velocity value from a cached waveform result.

    If native_baseline_global is provided, the function applies a direction-aware
    first-order baseline re-reference. Without that argument, it reproduces the
    existing auto-baseline behavior.
    """
    if waveform_result is None:
        return {
            "ok": False,
            "max_velocity_px": np.nan,
            "max_velocity_cm_s": np.nan,
            "baseline_shift_px": np.nan,
        }

    velocity_smooth = np.asarray(waveform_result["velocity_smooth"], dtype=float)

    if len(velocity_smooth) == 0:
        return {
            "ok": False,
            "max_velocity_px": np.nan,
            "max_velocity_cm_s": np.nan,
            "baseline_shift_px": np.nan,
        }

    max_velocity_px = float(np.nanmax(velocity_smooth))
    baseline_shift_px = 0.0

    if native_baseline_global is not None and np.isfinite(max_velocity_px):
        auto_baseline_roi = float(waveform_result["baseline"])
        native_baseline_roi = float(native_baseline_global) - float(roi_y_min)
        direction = waveform_result.get("direction", "unknown")

        if direction == "above":
            baseline_shift_px = native_baseline_roi - auto_baseline_roi
        elif direction == "below":
            baseline_shift_px = auto_baseline_roi - native_baseline_roi
        else:
            baseline_shift_px = 0.0

        max_velocity_px = max_velocity_px + baseline_shift_px

    return {
        "ok": True,
        "max_velocity_px": max_velocity_px,
        "max_velocity_cm_s": max_velocity_px * float(cm_s_per_px),
        "baseline_shift_px": baseline_shift_px,
    }