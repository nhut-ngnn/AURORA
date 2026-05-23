import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .Cross_model import CrossModalEncoders
from .AudioGuided_GMU import AudioGuidedGatedFusion
from .Classifier import MLPClassifier


class RepairMLP(nn.Module):
    """
    Residual correction in text embedding space:
      delta_raw = MLP([z_audio; z_asr])  -> (B, D)
      delta     = delta_scale * tanh(delta_raw)  (bounded)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class UncertaintyGate(nn.Module):
    """
    alpha = sigmoid(MLP([z_audio; z_asr])) in (0,1), then clamped to avoid saturation.
    """
    def __init__(
        self,
        audio_dim: int,
        asr_dim: Optional[int],
        hidden_dim: int,
        dropout: float = 0.1,
        use_asr: bool = True,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
    ):
        super().__init__()
        self.use_asr = use_asr
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        gate_in = audio_dim + (asr_dim if (use_asr and asr_dim is not None) else 0)
        self.net = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_audio: torch.Tensor, z_asr: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate_input = z_audio
        if self.use_asr and z_asr is not None:
            gate_input = torch.cat([z_audio, z_asr], dim=-1)

        alpha = torch.sigmoid(self.net(gate_input))  
        alpha = alpha.clamp(self.alpha_min, self.alpha_max)
        return alpha


class AURORA(nn.Module):
    def __init__(
        self,
        text_input_dim: int = 768,
        audio_input_dim: int = 768,
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        linear_layer_dims=None,
        num_classes: int = 4,

        repair_hidden_dim: int = 256,   
        gate_hidden_dim: int = 128,
        use_asr_in_gate: bool = True,

        delta_scale: float = 0.5,         
        alpha_min: float = 0.05,        
        alpha_max: float = 0.95,
    ):
        super().__init__()
        if linear_layer_dims is None:
            linear_layer_dims = [512, 128]

        self.fusion_dim = fusion_dim
        self.delta_scale = delta_scale

        self.encoders = CrossModalEncoders(
            text_input_dim, audio_input_dim, fusion_dim, dropout, num_heads
        )

        self.repair = RepairMLP(
            input_dim=fusion_dim * 2,     # [z_audio; z_asr]
            hidden_dim=repair_hidden_dim,
            output_dim=fusion_dim,
            dropout=dropout,
        )

        self.uncertainty_gate = UncertaintyGate(
            audio_dim=fusion_dim,
            asr_dim=fusion_dim,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
            use_asr=use_asr_in_gate,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        self.gmu = AudioGuidedGatedFusion(
            text_dim=fusion_dim, audio_dim=fusion_dim, fusion_dim=fusion_dim
        )

        self.classifier = MLPClassifier(
            input_dim=fusion_dim,
            layer_dims=linear_layer_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _encode(self, text_feat: torch.Tensor, audio_feat: torch.Tensor):
        text_attn, audio_attn = self.encoders(text_feat, audio_feat)

        z_text = text_attn.mean(dim=1)   # (B,D)
        z_audio = audio_attn.mean(dim=1) # (B,D)
        return z_text, z_audio

    def _forward_student(self, text_asr: torch.Tensor, audio: torch.Tensor):
        z_asr, z_audio = self._encode(text_asr, audio)  

        repair_in = torch.cat([z_audio, z_asr], dim=-1)  # (B,2D)
        delta_raw = self.repair(repair_in)               # (B,D)

        delta = self.delta_scale * torch.tanh(delta_raw) # (B,D)

        alpha = self.uncertainty_gate(z_audio, z_asr)    # (B,1), clamped

        z_fused_text = z_asr + alpha * delta

        rep = self.gmu(z_fused_text, z_audio)
        logits = self.classifier(rep)

        return {
            "logits_student": logits,
            "student_rep": rep,
            "z_asr_text": z_asr,
            "z_audio_student": z_audio,
            "delta_raw": delta_raw,
            "delta": delta,
            "z_fused": z_fused_text,
            "alpha": alpha,
            "z_hallucinated": delta,
        }

    def _forward_teacher(self, text_clean: torch.Tensor, audio: torch.Tensor):
        z_clean, z_audio_t = self._encode(text_clean, audio)
        rep = self.gmu(z_clean, z_audio_t)
        logits = self.classifier(rep)
        return {
            "logits_teacher": logits,
            "teacher_rep": rep,
            "z_clean_text": z_clean,
            "z_audio_teacher": z_audio_t,
        }

    def forward(
        self,
        text_asr: torch.Tensor,
        audio: torch.Tensor,
        clean_text: Optional[torch.Tensor] = None,
        mode: str = "student",
        return_all: bool = True,
    ):
        outputs: Dict[str, Any] = {}

        if mode in ["student", "both"]:
            outputs.update(self._forward_student(text_asr, audio))

        if mode in ["teacher", "both"]:
            text_clean = clean_text if clean_text is not None else text_asr
            outputs.update(self._forward_teacher(text_clean, audio))

        if "logits_student" in outputs:
            outputs["logits"] = outputs["logits_student"]
        elif "logits_teacher" in outputs:
            outputs["logits"] = outputs["logits_teacher"]

        if return_all:
            return outputs

        if mode == "teacher":
            return outputs["logits_teacher"]
        return outputs["logits_student"]
