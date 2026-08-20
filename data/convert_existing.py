import os
import csv
import subprocess

RAW_DIR = "data/raw"
CSV_URL_PATH = "data/audiocaps_train.csv"

# map file index (from filename) to caption using the original audiocaps csv
rows = []
with open(CSV_URL_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        caption = row["caption"].lower()
        if any(k in caption for k in ["footstep", "door", "rain", "glass", "wind"]):
            rows.append(row)
rows = rows[:200]

metadata = []
for i, row in enumerate(rows):
    caption = row["caption"]
    src_webm = os.path.join(RAW_DIR, f"tmp_{i}.webm")
    src_m4a = os.path.join(RAW_DIR, f"tmp_{i}.m4a")
    src = src_webm if os.path.exists(src_webm) else (src_m4a if os.path.exists(src_m4a) else None)

    if src is None:
        continue

    out_file = f"clip_{i}.wav"
    out_path = os.path.join(RAW_DIR, out_file)

    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", src,
            "-ss", "0", "-t", "10",
            "-ar", "24000", "-ac", "1",
            out_path
        ], check=True, capture_output=True, timeout=30)
        metadata.append((caption, out_file))
        print(f"[{i}] converted: {caption}")
    except Exception as e:
        print(f"[{i}] failed: {e}")

with open("data/metadata.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["caption", "audio_filename"])
    for caption, filename in metadata:
        writer.writerow([caption, filename])

print(f"Total converted: {len(metadata)}")