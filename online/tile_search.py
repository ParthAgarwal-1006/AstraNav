import cv2


class TileSearcher:

    def __init__(self, database, metadata):

        self.database = database
        self.metadata = metadata

        self.tile_names = [
            item["tile"]
            for item in metadata
        ]

        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    # -------------------------------------------------
    # Candidate tiles using predicted position
    # -------------------------------------------------

    def get_candidate_tiles(self, predicted_position):

        if predicted_position is None:
            return self.database

        predicted_x = predicted_position[0]

        predicted_tile = None

        for item in self.metadata:

            left = item["x"]
            right = left + 512

            if left <= predicted_x < right:

                predicted_tile = item["tile"]
                break

        if predicted_tile is None:

            return self.database

        idx = self.tile_names.index(predicted_tile)

        start = max(0, idx - 1)
        end = min(len(self.database), idx + 2)

        return self.database[start:end]

    # -------------------------------------------------
    # Search best tile
    # -------------------------------------------------

    def search(

        self,
        kp_drone,
        des_drone,
        candidate_tiles,

    ):

        best_tile = None
        best_matches = []
        best_kp_tile = None

        for item in candidate_tiles:

            if item["descriptors"] is None:
                continue

            kp_tile = [

                cv2.KeyPoint(

                    x=float(k[0]),
                    y=float(k[1]),
                    size=float(k[2]),
                    angle=float(k[3]),
                    response=float(k[4]),
                    octave=int(k[5]),
                    class_id=int(k[6]),

                )

                for k in item["keypoints"]

            ]

            matches = self.matcher.knnMatch(

                des_drone,
                item["descriptors"],
                k=2,

            )

            good = []

            for m, n in matches:

                if m.distance < 0.75 * n.distance:

                    good.append(m)

            if len(good) > len(best_matches):

                best_matches = good
                best_tile = item["tile"]
                best_kp_tile = kp_tile

        return (

            best_tile,
            best_kp_tile,
            best_matches,

        )
    