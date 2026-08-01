import cv2
import pandas as pd
from pathlib import Path


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

satellite_path = Path(
    "datasets/satellite/satellite.png"
)

predictions_path = Path(
    "outputs/localization/predictions.csv"
)

output_path = Path(
    "outputs/trajectory/trajectory_result.png"
)


# ---------------------------------------------------
# Load satellite image and predictions
# ---------------------------------------------------

satellite = cv2.imread(str(satellite_path))

predictions = pd.read_csv(predictions_path)

if satellite is None:
    raise FileNotFoundError(
        f"Satellite image not found: {satellite_path}"
    )


# ---------------------------------------------------
# Draw complete trajectory
# ---------------------------------------------------

trajectory = []

for _, row in predictions.iterrows():

    x = int(row["x"])
    y = int(row["y"])

    trajectory.append((x, y))


# Draw the complete path

for i in range(1, len(trajectory)):

    cv2.line(
        satellite,
        trajectory[i - 1],
        trajectory[i],
        (0, 255, 0),
        3,
    )


# Mark starting position

if len(trajectory) > 0:

    cv2.circle(
        satellite,
        trajectory[0],
        10,
        (255, 0, 0),
        -1,
    )

    cv2.putText(
        satellite,
        "Start",
        (
            trajectory[0][0] + 15,
            trajectory[0][1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


# Mark final position

if len(trajectory) > 0:

    cv2.circle(
        satellite,
        trajectory[-1],
        10,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        satellite,
        "End",
        (
            trajectory[-1][0] + 15,
            trajectory[-1][1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


# ---------------------------------------------------
# Add title
# ---------------------------------------------------

cv2.putText(
    satellite,
    "Estimated UAV Trajectory",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
)


# ---------------------------------------------------
# Save image
# ---------------------------------------------------

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

cv2.imwrite(
    str(output_path),
    satellite,
)

print(
    f"Trajectory image saved to: {output_path}"
)