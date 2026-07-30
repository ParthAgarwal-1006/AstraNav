import csv
from pathlib import Path

import cv2
import numpy as np


from online.kalman_filter import KalmanFilter2D
from online.pose_estimation import estimate_pose
from online.tile_search import TileSearcher

# ==========================================================
# Load Database
# ==========================================================

DATABASE = np.load(
    "outputs/database/database.npy",
    allow_pickle=True,
)

METADATA = np.load(
    "outputs/database/tile_metadata.npy",
    allow_pickle=True,
)

tile_lookup = {
    item["tile"]: (item["x"], item["y"])
    for item in METADATA
}

# ==========================================================
# Tile Search
# ==========================================================

searcher = TileSearcher(
    DATABASE,
    METADATA,
)

# ==========================================================
# Feature Extractor
# ==========================================================

sift = cv2.SIFT_create(
    nfeatures=4000
)

# ==========================================================
# Filters
# ==========================================================



kf = KalmanFilter2D()

kalman_initialized = False

VISION_INTERVAL = 1

# ==========================================================
# Frames
# ==========================================================

frames = sorted(
    Path("datasets/drone").glob("*.png")
)

print("=" * 60)
print("Frames Found :", len(frames))
print("=" * 60)

# ==========================================================
# Output CSV
# ==========================================================

Path("outputs/localization").mkdir(
    parents=True,
    exist_ok=True,
)

csv_file = open(
    "outputs/localization/predictions.csv",
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow(
    [
        "frame",
        "x",
        "y",
        "tile",
        "matches",
        "inliers",
        "reprojection_error",
        "confidence",
    ]
)

# ==========================================================
# Main Loop
# ==========================================================

for frame_path in frames:

    print(f"\nProcessing {frame_path.name}")

    frame_number = int(
        frame_path.stem.split("_")[1]
    )

    drone = cv2.imread(str(frame_path))

    drone_gray = cv2.cvtColor(
        drone,
        cv2.COLOR_BGR2GRAY,
    )

    kp_drone, des_drone = sift.detectAndCompute(
        drone_gray,
        None,
    )

    # ------------------------------------------------------
    # Kalman Prediction
    # ------------------------------------------------------

    predicted_position = None

    if kalman_initialized:

        predicted_position = kf.predict()

        print(
            "Predicted Position:",
            predicted_position,
        )

    # ------------------------------------------------------
    # Candidate Tile Search
    # ------------------------------------------------------

    candidate_tiles = searcher.get_candidate_tiles(
        predicted_position
    )

    print(
        "Candidate Tiles:",
        [x["tile"] for x in candidate_tiles],
    )

    # ------------------------------------------------------
    # Local Search
    # ------------------------------------------------------

    best_tile, best_kp_tile, best_matches = searcher.search(

        kp_drone,
        des_drone,
        candidate_tiles,

    )

    # ------------------------------------------------------
    # Global Search Fallback
    # ------------------------------------------------------

    if len(best_matches) < 80:

        print("Weak localization.")
        print("Searching all tiles...")

        best_tile, best_kp_tile, best_matches = searcher.search(

            kp_drone,
            des_drone,
            DATABASE,

        )

    if best_tile is None:

        print("Localization failed.")

        continue

    tile_x, tile_y = tile_lookup[best_tile]

    print("Best Tile :", best_tile)
    print("Matches   :", len(best_matches))

        # ------------------------------------------------------
    # Homography Estimation
    # ------------------------------------------------------

    H, mask, reproj = estimate_pose(

        kp_drone,
        best_kp_tile,
        best_matches,

    )

    if H is None:

        print("Homography failed.")

        continue

    # ------------------------------------------------------
    # UAV Center inside Tile
    # ------------------------------------------------------

    h, w = drone.shape[:2]

    corners = np.float32(

        [

            [0, 0],
            [w, 0],
            [w, h],
            [0, h],

        ]

    ).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(

        corners,
        H,

    )

    # ------------------------------------------------------
    # Convert Tile Coordinates
    # into Global Coordinates
    # ------------------------------------------------------

    projected[:, :, 0] += tile_x
    projected[:, :, 1] += tile_y

    vision_center = projected.mean(axis=0)[0]

    print("Vision :", vision_center)

    # ------------------------------------------------------
    # Confidence
    # ------------------------------------------------------

    inliers = 0

    if mask is not None:

        inliers = int(mask.sum())

    confidence = (

        inliers /
        max(len(best_matches), 1)

    )

    if confidence < 0.35:

        print("Weak homography rejected.")

        continue



    # ------------------------------------------------------
    # Kalman Fusion
    # ------------------------------------------------------

    if not kalman_initialized:

        kf.initialize(

            vision_center[0],
            vision_center[1],

        )

        kalman_initialized = True

        fused_position = vision_center.copy()

        print("Kalman Initialized")

    else:

        if frame_number % VISION_INTERVAL == 0:

            print("VISION UPDATE")

            fused_position = kf.update(
                vision_center
            )

        else:

            print("KALMAN PREDICTION")

            fused_position = predicted_position.copy()

            kf.x[0, 0] = fused_position[0]
            kf.x[1, 0] = fused_position[1]

    print("Fused :", fused_position)

    # ------------------------------------------------------
    # Save Prediction
    # ------------------------------------------------------

    writer.writerow(

        [

            frame_path.name,

            float(fused_position[0]),
            float(fused_position[1]),

            best_tile,

            len(best_matches),

            inliers,

            float(reproj),

            float(confidence),

        ]

    )

    # ------------------------------------------------------
    # Console Summary
    # ------------------------------------------------------

    print("-" * 60)

    print(f"Frame      : {frame_path.name}")
    print(f"Tile       : {best_tile}")
    print(f"Matches    : {len(best_matches)}")
    print(f"Inliers    : {inliers}")
    print(f"Confidence : {confidence:.2f}")
    print(
        f"Position   : ({fused_position[0]:.2f}, {fused_position[1]:.2f})"
    )

    print("-" * 60)

# ==========================================================
# Finish
# ==========================================================

csv_file.close()

print("\n" + "=" * 60)
print("Flight localization complete.")
print("Predictions saved to:")
print("outputs/localization/predictions.csv")
print("=" * 60)