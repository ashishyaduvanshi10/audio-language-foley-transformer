import torch
import torch.nn as nn

class FoleyTransformer(nn.Module):
    def __init__(self, text_embed_dim=512, codebook_size=1024, num_codebooks=8,
                 d_model=256, nhead=8, num_layers=4, max_seq_len=300):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # project text embedding into model dimension (used as memory for cross-attention)
        self.text_proj = nn.Linear(text_embed_dim, d_model)

        # token embedding for audio codes (sum across codebooks for simplicity)
        self.token_embed = nn.Embedding(codebook_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, codebook_size)

    def forward(self, text_embed, token_seq):
        # text_embed: [batch, text_embed_dim]
        # token_seq: [batch, seq_len] (first codebook only, simplified)
        memory = self.text_proj(text_embed).unsqueeze(1)  # [batch, 1, d_model]

        x = self.token_embed(token_seq) + self.pos_embed[:, :token_seq.size(1), :]

        seq_len = token_seq.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(token_seq.device)

        out = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
        logits = self.output_head(out)  # [batch, seq_len, codebook_size]
        return logits


if __name__ == "__main__":
    model = FoleyTransformer()
    text_embed = torch.randn(2, 512)
    token_seq = torch.randint(0, 1024, (2, 50))
    logits = model(text_embed, token_seq)
    print("Logits shape:", logits.shape)