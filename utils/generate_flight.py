import cv2
import numpy as np
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------

SATELLITE = "datasets/satellite/satellite.png"

OUTPUT = Path("datasets/drone")
OUTPUT.mkdir(parents=True, exist_ok=True)

x = 100
y = 250

crop_size = 350

step = 40

num_frames = 37

# -------------------------------

img = cv2.imread(SATELLITE)

for i in range(num_frames):

    xx = x + i * step

    crop = img[
        y:y + crop_size,
        xx:xx + crop_size
    ].copy()

    h, w = crop.shape[:2]

    # -----------------------------------
    # Random Rotation
    # -----------------------------------

    angle = np.random.uniform(-15, 15)

    M = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0,
    )

    crop = cv2.warpAffine(
        crop,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT,
    )

    # -----------------------------------
    # Random Brightness
    # -----------------------------------

    alpha = np.random.uniform(
        0.8,
        1.2,
    )

    beta = np.random.randint(
        -20,
        20,
    )

    crop = cv2.convertScaleAbs(
        crop,
        alpha=alpha,
        beta=beta,
    )

    # -----------------------------------
    # Gaussian Blur
    # -----------------------------------

    if np.random.rand() < 0.5:

        crop = cv2.GaussianBlur(
            crop,
            (5, 5),
            0,
        )

    # -----------------------------------
    # Noise
    # -----------------------------------

    noise = np.random.normal(
        0,
        5,
        crop.shape,
    ).astype(np.int16)

    crop = crop.astype(np.int16)

    crop += noise

    crop = np.clip(
        crop,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        OUTPUT / f"frame_{i:04d}.png",
        crop,
    )

print("=" * 50)
print("Synthetic UAV flight generated.")
print(f"Frames : {num_frames}")
print("=" * 50)