import torch
import torch.nn as nn

class CrossModalEncoders(nn.Module):
    def __init__(self, text_input_dim, audio_input_dim, fusion_dim, dropout, num_heads):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_input_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout)
        )

        self.cross_attention_text = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_audio = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.res_proj = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, text_feat, audio_feat):
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)

        text_encoded = self.text_encoder(text_feat)
        audio_encoded = self.audio_encoder(audio_feat)

        text_attn, _ = self.cross_attention_text(text_encoded, audio_encoded, audio_encoded)
        audio_attn, _ = self.cross_attention_audio(audio_encoded, text_encoded, text_encoded)
        
        text_out = self.res_proj(text_encoded) + text_attn
        audio_out = self.res_proj(audio_encoded) + audio_attn

        return text_out, audio_out