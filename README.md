
# AstraNav

## GPS-Denied UAV Localization Using Satellite Imagery

AstraNav is a vision-based UAV localization system designed for GPS-denied environments. The system estimates the UAV's position by matching aerial/drone images with a preprocessed satellite map.

The project uses **SIFT feature matching**, **RANSAC-based homography estimation**, **satellite tile search**, and **Kalman filtering** to estimate and track the UAV's position across a sequence of flight images.

---

## Project Pipeline

```text
Satellite Image
      ↓
Satellite Tiling
      ↓
SIFT Feature Extraction
      ↓
Offline Feature Database
      ↓
Drone Flight Frames
      ↓
SIFT Feature Extraction
      ↓
Candidate Tile Search
      ↓
Feature Matching
      ↓
Homography and RANSAC
      ↓
Vision-Based Position Estimation
      ↓
Kalman Prediction and Vision Correction
      ↓
Final UAV Trajectory
````

---

## Features

* GPS-denied UAV localization
* Satellite image divided into searchable map tiles
* Offline SIFT feature extraction
* Feature database for faster online localization
* Candidate tile search
* SIFT descriptor matching
* RANSAC-based outlier rejection
* Homography-based position estimation
* Kalman filtering for position tracking
* Periodic vision correction
* Flight trajectory visualization
* Simulated IMU integration module

---

## Project Structure

```text
AstraNav/
│
├── datasets/
│   ├── satellite/
│   └── drone/
│
├── offline/
│   ├── 01_tile_satellite.py
│   ├── 02_extract_sift.py
│   └── 03_build_database.py
│
├── online/
│   ├── __init__.py
│   ├── imu.py
│   ├── kalman_filter.py
│   ├── pose_estimation.py
│   ├── run_flight.py
│   └── tile_search.py
│
├── outputs/
│   ├── database/
│   ├── features/
│   ├── localization/
│   ├── tiles/
│   └── trajectory/
│
├── utils/
│   ├── create_drone_crop.py
│   └── generate_flight.py
│
├── visualization/
│   └── create_animation.py
│
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Offline Satellite Preprocessing

The satellite image is divided into smaller tiles.

SIFT features and descriptors are extracted from every satellite tile. The extracted features are stored in a database so that expensive feature processing can be performed before the UAV flight.

This reduces the computational workload during online localization.

### 2. Online UAV Localization

For every drone frame:

1. SIFT keypoints and descriptors are extracted.
2. Candidate satellite tiles are selected using the predicted UAV position.
3. The drone features are matched with the candidate tile features.
4. The best matching satellite tile is selected.
5. RANSAC removes incorrect feature matches.
6. A homography is estimated between the drone image and the satellite tile.
7. The drone image is projected onto the satellite map.
8. The center of the projected image is used as the vision-based UAV position.

### 3. Kalman Filtering

The Kalman filter maintains an estimated UAV position and predicts the next position.

Vision-based localization is used periodically to correct accumulated prediction error.

This provides a smoother trajectory than using individual image-localization results independently.

### 4. IMU Integration

The project includes an IMU integration module that maintains an estimated position and velocity.

In a real UAV, the IMU would provide acceleration and angular-rate measurements at a high frequency. The IMU would propagate the UAV state between camera frames, while satellite-image localization would periodically correct the accumulated inertial drift.

The current implementation provides the software architecture for this integration. Real IMU validation requires synchronized sensor data and UAV hardware.

---

## Why SIFT Was Used

SIFT was selected because it provides:

* Scale-invariant feature detection
* Rotation robustness
* Strong local feature descriptors
* Reliable matching between aerial images
* Compatibility with RANSAC and homography estimation

SIFT is computationally heavier than lightweight learned feature models, but it was suitable for the initial prototype because it provides an interpretable and reliable classical baseline.

---

## Future Improvements

* Integrate real IMU measurements
* Add camera calibration and camera-to-body extrinsic calibration
* Use Visual-Inertial Odometry
* Use an Extended Kalman Filter or factor-graph optimization
* Add UAV altitude and orientation information
* Improve candidate-tile selection
* Use hierarchical map search
* Replace SIFT with learned features such as SuperPoint
* Use learned matching methods such as LightGlue or LoFTR
* Use DINOv2 for global image retrieval and candidate-tile selection
* Use GPU acceleration
* Evaluate localization accuracy using a larger dataset
* Test with real UAV flight data

---

## Results

The system generates:

* UAV position predictions:

```text
outputs/localization/predictions.csv
```

* Satellite localization visualization:

```text
outputs/localization/satellite_result.png
```

* UAV trajectory video:

```text
outputs/trajectory/trajectory.mp4
```

---

## Requirements

* Python 3.10
* OpenCV
* NumPy
* Pandas

Install the dependencies using:

```bash
pip install opencv-python numpy pandas
```

---

## Running the Project

### Step 1: Generate satellite tiles

```bash
python offline/01_tile_satellite.py
```

### Step 2: Extract SIFT features

```bash
python offline/02_extract_sift.py
```

### Step 3: Build the feature database

```bash
python offline/03_build_database.py
```

### Step 4: Run UAV localization

```bash
python online/run_flight.py
```

### Step 5: Create the trajectory animation

```bash
python visualization/create_animation.py
```

---

## Output

The final output is a trajectory visualization showing:

* The estimated UAV position
* The UAV path over the satellite map
* The corresponding drone frame

---

## Author

Parth Agarwal

B.Tech Computer Science, Artificial Intelligence and Machine Learning
UPES, Dehradun

```

