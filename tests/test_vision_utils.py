import numpy as np
from BYS_XAISwallow.vision_utils import crop_square

def test_crop_square_centered():
    img = np.zeros((100, 120, 3), np.uint8)
    roi, box = crop_square(img, (60, 50), 40)
    assert roi.shape == (40, 40, 3)
    x1, y1, x2, y2 = box
    assert x2 - x1 <= 40 and y2 - y1 <= 40