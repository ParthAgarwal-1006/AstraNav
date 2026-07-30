import numpy as np


class KalmanFilter2D:

    def __init__(self):

        # State:
        # [x, y, vx, vy]

        self.dt = 1.0

        self.x = np.zeros((4, 1), dtype=np.float32)

        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.P = np.eye(4, dtype=np.float32) * 100

        self.Q = np.array([
            [0.25, 0, 0.5, 0],
            [0, 0.25, 0, 0.5],
            [0.5, 0, 1.0, 0],
            [0, 0.5, 0, 1.0]
        ], dtype=np.float32) * 0.05

        self.R = np.eye(2, dtype=np.float32) * 16

        self.initialized = False

        self.last_measurement = None

    # ---------------------------------------

    def initialize(self, x, y):

        self.x = np.array([
            [x],
            [y],
            [0],
            [0]
        ], dtype=np.float32)

        self.last_measurement = np.array([x, y], dtype=np.float32)

        self.initialized = True

    # ---------------------------------------

    def predict(self):

        self.x = self.A @ self.x

        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.position

    # ---------------------------------------

    def update(self, measurement):

        measurement = np.asarray(
            measurement,
            dtype=np.float32
        )

        if self.last_measurement is not None:

            measured_velocity = (
                measurement - self.last_measurement
            )

            self.x[2, 0] = (
                0.8 * self.x[2, 0]
                + 0.2 * measured_velocity[0]
            )

            self.x[3, 0] = (
                0.8 * self.x[3, 0]
                + 0.2 * measured_velocity[1]
            )

        self.last_measurement = measurement.copy()

        z = measurement.reshape(2, 1)

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        self.P = (
            np.eye(4, dtype=np.float32)
            - K @ self.H
        ) @ self.P

        return self.position

    # ---------------------------------------

    @property
    def position(self):

        return self.x[:2].flatten()

    @property
    def velocity(self):

        return self.x[2:].flatten()