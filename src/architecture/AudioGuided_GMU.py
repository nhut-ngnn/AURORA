import torch.nn as nn

class AudioGuidedGatedFusion(nn.Module):
    def __init__(self, text_dim, audio_dim, fusion_dim, dropout_p=0.2):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.gate_audio = nn.Linear(fusion_dim, fusion_dim)
        self.gate_fusion = nn.Linear(fusion_dim, fusion_dim)

        self.out_proj = nn.Linear(fusion_dim, fusion_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, text_feat, audio_feat):
        f_t = self.text_proj(text_feat)
        f_a = self.audio_proj(audio_feat)

        f_at = f_a + f_t

        f_at_prime = self.fusion_ffn(f_at) + f_at

        gate_pre = self.gate_audio(f_a) + self.gate_fusion(f_at_prime)
        w = self.sigmoid(gate_pre)

        fused = (1.0 - w) * f_a + w * f_at_prime
        fused = self.out_proj(fused)
        return fused
