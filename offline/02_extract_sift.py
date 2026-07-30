import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


feature_dir = Path("outputs/features")
feature_dir.mkdir(parents=True, exist_ok=True)

for file in feature_dir.glob("*.npz"):
    file.unlink()
TILES = Path("outputs/tiles")
OUT = Path("outputs/features")

OUT.mkdir(parents=True, exist_ok=True)

sift = cv2.SIFT_create(nfeatures=4000)

tile_paths = sorted(TILES.glob("*.png"))

print(f"Found {len(tile_paths)} tiles")

for tile_path in tqdm(tile_paths):

    image = cv2.imread(str(tile_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)

    kp_array = np.array(
        [
            (
                kp.pt[0],
                kp.pt[1],
                kp.size,
                kp.angle,
                kp.response,
                kp.octave,
                kp.class_id,
            )
            for kp in keypoints
        ],
        dtype=np.float32,
    )

    np.savez_compressed(
        OUT / f"{tile_path.stem}.npz",
        keypoints=kp_array,
        descriptors=descriptors,
    )

print("Finished extracting SIFT features.")