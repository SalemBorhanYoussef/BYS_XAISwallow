# yolo_pose_utils.py – YOLO-Pose Inferenz + Overlay
from typing import Tuple, Optional, Dict, Any
import numpy as np, cv2
from ultralytics import YOLO

def run_yolo_pose(model: YOLO, bgr: np.ndarray, conf: float = 0.25, device: str = "cpu"
                  ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]], np.ndarray]:
    """
    Returns: (keypoints_best[K,3], box_best{ x1,y1,x2,y2 }, annotated_bgr)
    """
    pred = model.predict(bgr, conf=conf, device=device, verbose=False)
    if not pred:
        return None, None, bgr
    r = pred[0]
    annotated = r.plot()  # BGR

    kp = None
    box = None
    if r.keypoints is not None and r.keypoints.data is not None and len(r.keypoints.data) > 0:
        kps = r.keypoints.data.cpu().numpy()  # (N,K,3)
        idx = 0
        if r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf.cpu().numpy()
            idx = int(np.argmax(confs))
            xyxy = r.boxes.xyxy.cpu().numpy()[idx]
            box = {"x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3])}
        kp = kps[idx]
    return kp, box, annotated
