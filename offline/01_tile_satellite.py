import cv2
import numpy as np
from pathlib import Path



tiles_dir = Path("outputs/tiles")
tiles_dir.mkdir(parents=True, exist_ok=True)

for file in tiles_dir.glob("*.png"):
    file.unlink()

SATELLITE = "datasets/satellite/satellite.png"

OUTPUT_TILE = Path("outputs/tiles")
OUTPUT_DB = Path("outputs/database")

OUTPUT_TILE.mkdir(parents=True, exist_ok=True)
OUTPUT_DB.mkdir(parents=True, exist_ok=True)

tile_size = 600
stride = 500

image = cv2.imread(SATELLITE)

H, W = image.shape[:2]

metadata = []

count = 0

for y in range(0, H - tile_size + 1, stride):

    for x in range(0, W - tile_size + 1, stride):

        tile = image[y:y + tile_size, x:x + tile_size]

        name = f"tile_{count:05d}"

        cv2.imwrite(
            str(OUTPUT_TILE / f"{name}.png"),
            tile,
        )

        metadata.append(
            {
                "tile": name,
                "x": x,
                "y": y,
            }
        )

        count += 1

np.save(
    OUTPUT_DB / "tile_metadata.npy",
    metadata,
)

print("\n==================================================")
print("Satellite tiled.")
print(f"Tiles : {count}")
print("Metadata saved.")
print("==================================================")