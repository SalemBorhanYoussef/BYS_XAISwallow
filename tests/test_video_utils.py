import os
import cv2
import numpy as np
import tempfile

from BYS_XAISwallow.video_utils import read_rgb_frames

def _make_tmp_video(path: str, size=(64, 96), frames=10, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (size[1], size[0]))
    assert vw.isOpened()
    for i in range(frames):
        img = np.zeros((size[0], size[1], 3), np.uint8)
        cv2.putText(img, f"{i}", (5, size[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()

def test_read_rgb_frames_basic():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.mp4")
        _make_tmp_video(path, size=(32, 48), frames=12)
        arr = read_rgb_frames(path, start=0, max_frames=5, stride=1)
        assert arr.shape[0] == 5
        assert arr.shape[1:] == (32, 48, 3)
        assert arr.dtype == np.uint8

def test_read_rgb_frames_stride_resize():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_stride.mp4")
        _make_tmp_video(path, size=(40, 60), frames=20)
        arr = read_rgb_frames(path, stride=2, resize_hw=(20, 30))
        # 20 frames stride 2 => ca. 10 behalten (EOF kann minimal abweichen) -> überprüfen >=9
        assert arr.shape[0] >= 9
        assert arr.shape[1:] == (20, 30, 3)
        assert arr.dtype == np.uint8
