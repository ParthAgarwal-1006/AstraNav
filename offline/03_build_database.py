import numpy as np
from pathlib import Path

FEATURES = Path("outputs/features")
DATABASE = Path("outputs/database")

DATABASE.mkdir(parents=True, exist_ok=True)

feature_files = sorted(FEATURES.glob("*.npz"))

database = []

print(f"Loading {len(feature_files)} feature files...")

for file in feature_files:

    data = np.load(file)

    database.append(
        {
            "tile": file.stem,
            "keypoints": data["keypoints"],
            "descriptors": data["descriptors"],
        }
    )

np.save(
    DATABASE / "database.npy",
    database,
    allow_pickle=True,
)

print()
print("=" * 50)
print("Database created successfully.")
print(f"Tiles stored : {len(database)}")
print("=" * 50)