import os
import csv
import subprocess
import urllib.request

KEYWORDS = ["footstep", "door", "rain", "glass", "wind"]
OUT_DIR = "data/raw"
CSV_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv"
MAX_CLIPS = 200

os.makedirs(OUT_DIR, exist_ok=True)

csv_path = "data/audiocaps_train.csv"
if not os.path.exists(csv_path):
    urllib.request.urlretrieve(CSV_URL, csv_path)

rows = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        caption = row["caption"].lower()
        if any(k in caption for k in KEYWORDS):
            rows.append(row)

rows = rows[:MAX_CLIPS]

metadata = []
for i, row in enumerate(rows):
    yt_id = row["youtube_id"]
    start = float(row["start_time"])
    caption = row["caption"]
    out_file = f"clip_{i}.wav"
    out_path = os.path.join(OUT_DIR, out_file)

    if os.path.exists(out_path):
        metadata.append((caption, out_file))
        continue

    url = f"https://www.youtube.com/watch?v={yt_id}"
    tmp = os.path.join(OUT_DIR, f"tmp_{i}.wav")

    try:
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "wav",
            "--postprocessor-args", f"-ss {start} -t 10",
            "-o", tmp, url
        ], check=True, timeout=60)

        if os.path.exists(tmp):
            os.rename(tmp, out_path)
            metadata.append((caption, out_file))
            print(f"[{i}] done: {caption}")
    except Exception as e:
        print(f"[{i}] skipped: {e}")

with open("data/metadata.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["caption", "audio_filename"])
    for caption, filename in metadata:
        writer.writerow([caption, filename])

print(f"Total clips downloaded: {len(metadata)}")