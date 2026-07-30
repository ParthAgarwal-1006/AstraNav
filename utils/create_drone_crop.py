import cv2
import numpy as np
from pathlib import Path

image = cv2.imread("datasets/satellite/satellite.png")

# Crop position
x = 400
y = 250
w = 350
h = 350

crop = image[y:y+h, x:x+w]

Path("datasets/drone").mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    "datasets/drone/frame_01.png",
    crop,
)

# Ground truth center
center_x = x + w / 2
center_y = y + h / 2

Path("outputs/localization").mkdir(parents=True, exist_ok=True)

np.save(
    "outputs/localization/ground_truth.npy",
    np.array([center_x, center_y]),
)

print("Drone image created.")
print(f"Top-left      : ({x}, {y})")
print(f"Ground Truth  : ({center_x:.1f}, {center_y:.1f})")