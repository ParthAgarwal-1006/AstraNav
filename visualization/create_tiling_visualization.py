import cv2
from pathlib import Path


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

satellite_path = Path(
    "datasets/satellite/satellite.png"
)

metadata_path = Path(
    "outputs/database/tile_metadata.npy"
)

output_path = Path(
    "outputs/tiles/satellite_tiling.png"
)


# ---------------------------------------------------
# Load satellite image
# ---------------------------------------------------

satellite = cv2.imread(
    str(satellite_path)
)

if satellite is None:
    raise FileNotFoundError(
        f"Satellite image not found: {satellite_path}"
    )


# ---------------------------------------------------
# Load tile metadata
# ---------------------------------------------------

import numpy as np

metadata = np.load(
    str(metadata_path),
    allow_pickle=True
)


# ---------------------------------------------------
# Draw tile boundaries
# ---------------------------------------------------

output = satellite.copy()

for item in metadata:

    tile_name = item["tile"]

    x = int(item["x"])
    y = int(item["y"])

    # Get the corresponding tile image

    tile_path = Path(
        "outputs/tiles"
    ) / f"{tile_name}.png"

    tile = cv2.imread(
        str(tile_path)
    )

    if tile is None:
        continue

    tile_h, tile_w = tile.shape[:2]

    # Draw tile boundary

    cv2.rectangle(
        output,
        (x, y),
        (x + tile_w, y + tile_h),
        (0, 255, 0),
        4,
    )

    # Add tile name

    cv2.putText(
        output,
        tile_name,
        (x + 15, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
    )


# ---------------------------------------------------
# Add title
# ---------------------------------------------------

cv2.putText(
    output,
    "Satellite Map Tiling",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    (255, 255, 255),
    3,
)


# ---------------------------------------------------
# Save
# ---------------------------------------------------

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

cv2.imwrite(
    str(output_path),
    output,
)

print(
    f"Tiling visualization saved to: {output_path}"
)