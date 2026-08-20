import os
import csv
import torch
import soundfile as sf
import torchaudio
from encodec import EncodecModel

SAMPLE_RATE = 24000
RAW_DIR = "data/raw"
TOKEN_DIR = "data/tokens"
METADATA_IN = "data/metadata.csv"
METADATA_OUT = "data/metadata_tokens.csv"

os.makedirs(TOKEN_DIR, exist_ok=True)

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.eval()

rows = []
with open(METADATA_IN, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

output_rows = []
for i, row in enumerate(rows):
    caption = row["caption"]
    filename = row["audio_filename"]
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        continue

    try:
        data, sr = sf.read(path)
        wav = torch.from_numpy(data).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)

        wav = wav.unsqueeze(0)
        with torch.no_grad():
            encoded = model.encode(wav)
        codes = torch.cat([e[0] for e in encoded], dim=-1)

        token_file = filename.replace(".wav", ".pt")
        torch.save(codes, os.path.join(TOKEN_DIR, token_file))
        output_rows.append((caption, token_file))
        print(f"[{i}] tokenized: {caption}")
    except Exception as e:
        print(f"[{i}] failed: {e}")

with open(METADATA_OUT, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["caption", "token_file"])
    for caption, token_file in output_rows:
        writer.writerow([caption, token_file])

print(f"Total tokenized: {len(output_rows)}")