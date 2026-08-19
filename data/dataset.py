import torch
from torch.utils.data import Dataset
import torchaudio
import os

class FoleyDataset(Dataset):
    def __init__(self, audio_dir, text_csv, sample_rate=24000, max_duration=5.0):
        """
        Dataset loader for Audio-Language Foley Transformer.
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        
        # Placeholder for text prompt + audio file mappings
        # (Will load paths and text captions from CSV/JSON metadata)
        self.data_samples = [] 

    def __len__(self):
        return len(self.data_samples)

    def _load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if audio sample rate doesn't match target rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Pad or truncate waveform to fixed max duration
        if waveform.shape[1] < self.max_samples:
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_samples]
            
        return waveform

    def __getitem__(self, idx):
        # Example sample schema: {"text": "footsteps on gravel", "audio_path": "path/to/file.wav"}
        sample = self.data_samples[idx]
        text_prompt = sample["text"]
        waveform = self._load_audio(os.path.join(self.audio_dir, sample["audio_path"]))
        
        return {
            "text": text_prompt,
            "waveform": waveform
        }