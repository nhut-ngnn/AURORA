import torch
import numpy as np
from src.feature_extract.config import TOKENIZER, TEXT_MODEL
from src.architecture.ASR_model import SpeechToText
from trainer.extract_feature import extract_text_features
import time
from fvcore.nn import FlopCountAnalysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_stt_model = None
def _get_stt(device):
    global _stt_model
    if _stt_model is None:
        _stt_model = SpeechToText(device=device)
    return _stt_model

def compute_asr_embeddings(audio_paths, cached_asr, text_ds, device, prefer_cached=True):
    cached = cached_asr.to(device) if cached_asr is not None else None
    num_items = len(audio_paths) if hasattr(audio_paths, "__len__") else 0
    if num_items == 0:
        return None

    def _has_content(tensor):
        return tensor is not None and torch.is_tensor(tensor) and tensor.numel() > 0 and torch.any(tensor != 0)

    if prefer_cached and cached is not None and cached.ndim == 2:
        if torch.all(cached.abs().sum(dim=1) > 0):
            return cached

    stt = _get_stt(device)
    TEXT_MODEL.to(device).eval()
    embeds = []

    for idx in range(num_items):
        cached_embed = cached[idx] if cached is not None else None
        if prefer_cached and _has_content(cached_embed):
            embeds.append(cached_embed)
            continue

        asr_embed = None
        path = audio_paths[idx]
        if isinstance(path, str) and path:
            text = stt.transcribe(path)
            asr_embed = extract_text_features(text, TOKENIZER, TEXT_MODEL, device)
            if asr_embed is not None:
                asr_embed = asr_embed.to(device)

        if asr_embed is None:
            if cached_embed is not None and cached_embed.numel() > 0:
                asr_embed = cached_embed
            else:
                ref = None
                if text_ds is not None and hasattr(text_ds, "__len__") and len(text_ds) > idx:
                    ref = text_ds[idx].to(device)
                elif cached is not None and hasattr(cached, "__len__") and len(cached) > idx:
                    ref = cached[idx]
                if ref is None:
                    continue
                asr_embed = torch.zeros_like(ref)

        embeds.append(asr_embed)

    if not embeds:
        return None
    return torch.stack(embeds)

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if classes.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    wa = float((y_true == y_pred).mean())
    recalls = []
    f1s = []
    supports = []

    for cls in classes:
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))

        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    ua = float(np.mean(recalls)) if recalls else 0.0
    support_sum = float(np.sum(supports))
    wf1 = float(np.sum(np.array(f1s) * np.array(supports)) / support_sum) if support_sum > 0 else 0.0
    uf1 = float(np.mean(f1s)) if f1s else 0.0
    return wa, ua, wf1, uf1

def evaluate_stage(model, loader, device, use_clean_text=False, return_all=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for audio, clean_text, asr_text, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            text_in = clean_text.to(device) if use_clean_text else asr_text.to(device)
            logits = model(text_asr=text_in, audio=audio, return_all=False)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    wa, ua, wf1, uf1 = compute_metrics(targets, preds)
    if return_all:
        return wa, ua, wf1, uf1
    return wa, ua

def get_model_stats(model, sample_input, device="cuda"):
    model.eval().to(device)

    flops = FlopCountAnalysis(model, sample_input).total()
    params = sum(p.numel() for p in model.parameters())

    n_runs = 50
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(*sample_input) if isinstance(sample_input, tuple) else model(sample_input)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_runs

    return {
        "FLOPs": flops,
        "Parameters": params,
        "Inference time (s)": avg_time
    }
