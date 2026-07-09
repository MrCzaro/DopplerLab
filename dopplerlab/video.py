__all__ = ['read_video_frame_by_index_v2']

# %% ../NB13 - Native-assisted AVI pipeline improvement validation.ipynb #dbf8569c-d982-437a-b066-680fe93eba6e
from pathlib import Path
import cv2


def read_video_frame_by_index_v2(video_path, frame_idx):
    """
    This function reads a video frame by frame index and returns frame timing information.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS detected for video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame_idx={frame_idx} from video: {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_time_s = int(frame_idx) / fps

    return {
        "frame_rgb": frame_rgb,
        "fps": float(fps),
        "frame_idx": int(frame_idx),
        "frame_time_s": float(frame_time_s),
    }