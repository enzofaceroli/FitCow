import os
from PIL import Image
import pandas as pd

# ======================================================
# CHANGE THESE PATHS
# ======================================================
INPUT_DATASET = "assets/cropped_dataset"
OUTPUT_DATASET = "assets/rotated_dataset"

IMAGE_EXTENSIONS = (".jpg")

# ======================================================
# CREATE OUTPUT FOLDERS
# ======================================================
os.makedirs(OUTPUT_DATASET, exist_ok=True)

for fold in os.listdir(INPUT_DATASET):
    fold_path = os.path.join(INPUT_DATASET, fold)

    if not os.path.isdir(fold_path):
        continue

    os.makedirs(os.path.join(OUTPUT_DATASET, fold), exist_ok=True)

# ======================================================
# ROTATE IMAGES
# ======================================================
angles = [0, 90, 180, 270]

for fold in os.listdir(INPUT_DATASET):

    input_fold = os.path.join(INPUT_DATASET, fold)

    if not os.path.isdir(input_fold):
        continue

    output_fold = os.path.join(OUTPUT_DATASET, fold)

    for filename in os.listdir(input_fold):

        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(input_fold, filename)
        with Image.open(image_path) as image:
            basename, ext = os.path.splitext(filename)

            for angle in angles:
                if angle == 0:
                    rotated = image
                else:
                    rotated = image.rotate(-angle, expand=False)

                rotated.save(
                    os.path.join(output_fold, f"{basename}_{angle}{ext}")
                )

# ======================================================
# CREATE NEW labels.csv
# ======================================================
csv_path = "assets/fitcow_label.csv"
output_csv = "assets/rotated_labels.csv"

print(f"Reading: {csv_path}")
print(f"Exists? {os.path.exists(csv_path)}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows.")

new_rows = []

for _, row in df.iterrows():
    basename, ext = os.path.splitext(row["file"])

    for angle in angles:
        new_rows.append({
            "file": f"{basename}_{angle}{ext}",
            "class": row["class"],
            "fold": row["fold"]
        })

new_df = pd.DataFrame(new_rows)

print(f"Saving {len(new_df)} rows to {output_csv}")
new_df.to_csv(output_csv, index=False)

print("CSV saved.")