import cv2
import numpy as np


def estimate_pose(kp_drone, kp_tile, matches):

    if len(matches) < 4:
        return None, None, None

    src_pts = np.float32(
        [kp_drone[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp_tile[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0,
    )

    if H is None:
        return None, None, None

    # -------------------------------
    # Compute reprojection error
    # -------------------------------

    projected = cv2.perspectiveTransform(
        src_pts,
        H,
    )

    error = np.linalg.norm(
        projected - dst_pts,
        axis=2,
    )

    error = float(error.mean())

    return H, mask, error