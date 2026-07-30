import numpy as np


class IMUSimulator:

    def __init__(self):

        self.position = None
        self.velocity = np.zeros(2, dtype=np.float32)

    def initialize(self, position):

        self.position = position.astype(np.float32)

    def predict(self):

        if self.position is None:
            return None

        self.position = self.position + self.velocity

        return self.position.copy()

    def update(self, vision_position):

        if self.position is None:

            self.initialize(vision_position)

            return vision_position

        alpha = 0.8

        fused = (
            alpha * vision_position
            + (1 - alpha) * self.position
        )

        self.velocity = fused - self.position

        self.position = fused

        return fused