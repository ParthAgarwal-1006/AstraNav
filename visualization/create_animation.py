import cv2
import pandas as pd
from pathlib import Path

satellite = cv2.imread("datasets/satellite/satellite.png")

pred = pd.read_csv("outputs/localization/predictions.csv")

height, width = satellite.shape[:2]

Path("outputs/trajectory").mkdir(parents=True, exist_ok=True)

video = cv2.VideoWriter(
    "outputs/trajectory/trajectory.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (width, height),
)

trajectory = []

for _, row in pred.iterrows():

    frame_name = row["frame"]

    x = int(row["x"])
    y = int(row["y"])

    # Skip impossible coordinates
    if x < 0 or x >= width or y < 0 or y >= height:
        print(f"Skipping {frame_name}: ({x}, {y}) outside map")
        continue

    img = satellite.copy()

    trajectory.append((x, y))

    # --------------------------
    # Draw trajectory
    # --------------------------

    if len(trajectory) > 1:

        for i in range(1, len(trajectory)):

            cv2.line(
                img,
                trajectory[i - 1],
                trajectory[i],
                (0, 255, 0),
                2,
            )

    # --------------------------
    # Draw current UAV
    # --------------------------

    cv2.circle(
        img,
        (x, y),
        8,
        (0, 0, 255),
        -1,
    )

    # --------------------------
    # Information
    # --------------------------

    cv2.putText(
        img,
        f"{frame_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        img,
        f"Tile : {row['tile']}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        img,
        f"Matches : {int(row['matches'])}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        img,
        f"Confidence : {row['confidence']:.2f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # --------------------------
    # Drone image
    # --------------------------

    drone_path = Path("datasets/drone") / frame_name

    if drone_path.exists():

        drone = cv2.imread(str(drone_path))

        drone = cv2.resize(drone, (220, 220))

        h, w = drone.shape[:2]

        img[20:20+h, width-w-20:width-20] = drone

    video.write(img)

video.release()

print("Trajectory video saved:")
print("outputs/trajectory/trajectory.mp4")