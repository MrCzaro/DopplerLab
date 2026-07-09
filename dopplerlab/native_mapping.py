__all__ = ['build_native_to_avi_mapping']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #02d578e3-bc27-486a-b8f4-46296013c87a
import pandas as pd
from pathlib import Path


def build_native_to_avi_mapping(native_batch_dir, avi_batch_dir):
    """
    This function maps native recording folders to AVI recording names by
    matching the byte size of each native linked AVI file to batch AVI files.
    """
    native_batch_dir = Path(native_batch_dir)
    avi_batch_dir = Path(avi_batch_dir)

    if not native_batch_dir.exists():
        raise FileNotFoundError(f"Native batch folder does not exist: {native_batch_dir}")
    if not avi_batch_dir.exists():
        raise FileNotFoundError(f"AVI batch folder does not exist: {avi_batch_dir}")

    avi_names_by_size = {}

    for avi_path in sorted(avi_batch_dir.glob("*.avi")):
        file_size = avi_path.stat().st_size
        avi_names_by_size.setdefault(file_size, []).append(avi_path.stem)

    rows = []

    for native_recording_dir in sorted(native_batch_dir.glob("*SMP")):
        linked_avi_paths = sorted((native_recording_dir / "linked_avi").glob("*.avi"))

        if len(linked_avi_paths) == 0:
            rows.append(
                {
                    "recording_id": native_recording_dir.name,
                    "linked_avi_name": None,
                    "linked_avi_size": None,
                    "nb04_recording_name": None,
                    "candidate_count": 0,
                    "match_status": "no_linked_avi",
                }
            )
            continue

        linked_avi_path = linked_avi_paths[0]
        linked_avi_size = linked_avi_path.stat().st_size
        candidate_names = avi_names_by_size.get(linked_avi_size, [])

        if len(candidate_names) == 1:
            match_status = "matched"
            nb04_recording_name = candidate_names[0]
        elif len(candidate_names) > 1:
            match_status = "duplicate_size_ambiguous"
            nb04_recording_name = None
        else:
            match_status = "no_size_match"
            nb04_recording_name = None

        rows.append(
            {
                "recording_id": native_recording_dir.name,
                "linked_avi_name": linked_avi_path.name,
                "linked_avi_size": linked_avi_size,
                "nb04_recording_name": nb04_recording_name,
                "candidate_count": len(candidate_names),
                "match_status": match_status,
            }
        )

    return pd.DataFrame(rows)