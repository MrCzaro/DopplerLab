__all__ = ['find_repo_root', 'dopplerlab_project_paths']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #db2c399b-c047-4d9d-863e-e686d6833571
from pathlib import Path


def find_repo_root(start=None, markers=("ultrasound_recordings", "feature_exports")):
    """
    This function resolves the DopplerLab repository root by walking upward
    from a starting directory until all marker paths are found.
    """
    if start is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start).resolve()

    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    marker_text = ", ".join(markers)
    raise FileNotFoundError(
        f"Could not locate DopplerLab repo root from {current}. "
        f"Required markers: {marker_text}"
    )


def dopplerlab_project_paths(root=None):
    """
    This function builds standard DopplerLab project paths used by
    validation notebooks.
    """
    repo_root = find_repo_root() if root is None else Path(root).resolve()

    return {
        "root": repo_root,
        "recordings_dir": repo_root / "ultrasound_recordings",
        "feature_exports_dir": repo_root / "feature_exports",
        "native_batch_dir": repo_root / "ultrasound_recordings" / "batch_2026_06_13_native",
        "avi_batch_dir": repo_root / "ultrasound_recordings" / "batch_2026_06_13",
        "nb04_v2_frame_csv": repo_root / "feature_exports" / "nb04_v2_batch_2026_06_13" / "nb04_v2_frame_level_velocity_features.csv",
    }