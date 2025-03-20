import numpy as np
import rawpy
import cv2


def readrawfile(filename) -> np.ndarray:
    with rawpy.imread(filename) as raw:
        raw_img = raw.postprocess(
            gamma=(1, 1),
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.ACES,
            highlight_mode=5,
            no_auto_bright=True,
            half_size=True,  # half size
        )
    raw_img = np.clip(raw_img / 65535, 0, 1).astype(np.float32)
    raw_img = cv2.resize(raw_img, (1500, 1000), interpolation=cv2.INTER_AREA)
    return raw_img