__all__ = ['parse_dcm_region_pw']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #d1813af7-40e8-48c8-b34e-237b0b034ca8
import re
from pathlib import Path


def parse_dcm_region_pw(native_dir):
    """
    This function reads the PW DcmRegionPara entry from a native Mindray export folder.

    The returned values are metadata-derived ROI and calibration inputs.
    They are not clinical measurements by themselves.
    """
    native_dir = Path(native_dir)
    dcm_region_path = native_dir / "DcmRegionPara.txt"

    if not dcm_region_path.exists():
        raise FileNotFoundError(f"DcmRegionPara.txt was not found: {dcm_region_path}")

    text = dcm_region_path.read_text(errors="replace")
    blocks = re.split(r"DATA_TREE_BEGIN=DcmRegion\d+", text)

    for block in blocks:
        key_values = dict(re.findall(r"(\w+)=([-\d.]+)", block))

        if key_values.get("DataType") == "3" and key_values.get("SpatialFormat") == "3":
            x0 = int(key_values["X0"])
            x1 = int(key_values["X1"])
            y0 = int(key_values["Y0"])
            y1 = int(key_values["Y1"])
            vir_y = int(key_values["VirY"])

            return {
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "vir_y": vir_y,
                "baseline_y_global": y0 + vir_y,
                "baseline_y_roi": vir_y,
                "cm_s_per_px": abs(float(key_values["PhyDeltaY"])),
                "s_per_px": abs(float(key_values["PhyDeltaX"])),
                "data_type": int(key_values["DataType"]),
                "spatial_format": int(key_values["SpatialFormat"]),
            }

    raise ValueError(f"No PW DcmRegion entry found in {dcm_region_path}")