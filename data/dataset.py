import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf
from transformers import AutoTokenizer

class FoleyDataset(Dataset):
    def __init__(self, metadata_csv, audio_dir, sample_rate=24000, max_duration=5.0):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        
        self.df = pd.read_csv(metadata_csv) if os.path.exists(metadata_csv) else pd.DataFrame()
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    def __len__(self):
        return len(self.df)

    def _load_audio(self, audio_path):
        # Load audio using soundfile directly (bypasses TorchCodec)
        data, sr = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()
        
        # Ensure 2D tensor format: [channels, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if sample rate doesn't match
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Pad or truncate waveform
        if waveform.shape[1] < self.max_samples:
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_samples]
            
        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_prompt = row["caption"]
        audio_path = os.path.join(self.audio_dir, row["audio_filename"])
        
        waveform = self._load_audio(audio_path)
        
        tokens = self.tokenizer(
            text_prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "text": text_prompt,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "waveform": waveform
        }