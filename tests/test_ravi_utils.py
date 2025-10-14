import numpy as np
from BYS_XAISwallow.ravi_utils import decode_ravi_frame

def test_decode_ravi_frame_basic():
    w, h = 8, 4
    # simulate packed bytes (uint8) representing uint16 pixels
    raw = (np.arange(w*h*2, dtype=np.uint8)).reshape(1, -1)
    arr = decode_ravi_frame(raw, w, h)
    assert arr.shape == (h, w)
    assert arr.dtype == np.uint16
