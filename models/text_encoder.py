import torch
import torch.nn as nn
from transformers import ClapTextModelWithProjection, AutoTokenizer

class CLAPTextEncoder(nn.Module):
    def __init__(self, model_name="laion/clap-htsat-unfused", freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ClapTextModelWithProjection.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, captions):
        tokens = self.tokenizer(
            captions, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(**tokens)
        return out.text_embeds

if __name__ == "__main__":
    encoder = CLAPTextEncoder()
    emb = encoder(["footsteps on gravel", "rain on tin roof"])
    print("Embedding shape:", emb.shape)