import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.dataloader import DATASET_CONFIG
from src.architecture.configs.asr_models import ASR_MODEL_CONFIGS, DEFAULT_ASR_MODEL_KEY
from src.training_process.data_io import build_feature_paths, resolve_checkpoint_dir, load_pkl
from src.architecture.AURORA import AURORA


def _normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def _prepare_words(text):
    cleaned = _normalize_text(text)
    return cleaned.split() if cleaned else []


def _levenshtein_distance(ref_words, hyp_words):
    m, n = len(ref_words), len(hyp_words)
    if m == 0:
        return None
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _compute_wer(clean_text, asr_text):
    ref_words = _prepare_words(clean_text)
    if not ref_words:
        return None, None
    hyp_words = _prepare_words(asr_text)
    distance = _levenshtein_distance(ref_words, hyp_words)
    if distance is None:
        return None, None
    return distance, len(ref_words)


def _to_tensor(value, device):
    tensor = torch.as_tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)

def _parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]

def _compute_metrics(y_true, y_pred):
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
    mf1 = float(np.mean(f1s)) if f1s else 0.0
    return wa, ua, wf1, mf1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict and show samples where ASR != clean text but model predicts correctly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="IEMOCAP",
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset name used to resolve feature prefix and class count.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="features_output",
        help="Directory containing extracted feature .pkl files.",
    )
    parser.add_argument(
        "--asr_model",
        type=str,
        default=DEFAULT_ASR_MODEL_KEY,
        choices=sorted(ASR_MODEL_CONFIGS.keys()),
        help="ASR model key used to build the feature directory and checkpoint path.",
    )
    parser.add_argument(
        "--use_clean_text_only",
        action="store_true",
        help="Use ground-truth text embeddings instead of ASR embeddings.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of classes (overridden by dataset config if present).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to select the default checkpoint name.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,52,103,128,923",
        help="Comma-separated seeds for evaluation averaging.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint (.pt).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Maximum samples to print; set to 0 to show all.",
    )
    parser.add_argument(
        "--min_wer",
        type=float,
        default=0.0,
        help="Minimum WER threshold to include a sample.",
    )
    parser.add_argument(
        "--sort_by_wer",
        action="store_true",
        help="Sort matched samples by WER descending before printing.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional CSV path to save matched samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset in DATASET_CONFIG:
        args.num_classes = DATASET_CONFIG[args.dataset]["num_classes"]

    cfg = DATASET_CONFIG[args.dataset]
    paths = build_feature_paths(args.data_dir, args.dataset, cfg["feature_prefix"], args.asr_model)
    test_path = paths["test"]
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test features not found: {test_path}")

    samples = load_pkl(test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wer_edits = 0
    wer_ref_words = 0
    for sample in samples:
        clean_text = sample.get("text_raw_dataset")
        asr_text = sample.get("text_raw_asr")
        if not clean_text or not asr_text:
            continue
        dist, ref_len = _compute_wer(clean_text, asr_text)
        if dist is None or not ref_len:
            continue
        wer_edits += dist
        wer_ref_words += ref_len
    wer = (wer_edits / wer_ref_words) if wer_ref_words > 0 else 0.0

    seeds = _parse_int_list(args.seeds)
    if args.checkpoint is not None:
        seeds = [args.seed]

    metrics = []
    for seed in seeds:
        checkpoint = args.checkpoint
        if checkpoint is None:
            ckpt_dir = resolve_checkpoint_dir("saved_model", args.dataset, args.asr_model)
            checkpoint = os.path.join(ckpt_dir, f"AURORA_{args.dataset}_seed{seed}.pt")
        if not os.path.exists(checkpoint):
            print(f"Checkpoint not found for seed {seed}: {checkpoint}")
            continue

        model = AURORA(num_classes=args.num_classes).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

        preds, targets = [], []
        for sample in samples:
            audio_embed = sample.get("audio_embed")
            asr_embed = sample.get("text_embed_asr")
            clean_embed = sample.get("text_embed_dataset")
            if asr_embed is None:
                asr_embed = sample.get("text_embed")
            if clean_embed is None:
                clean_embed = sample.get("text_embed")
            if audio_embed is None or asr_embed is None:
                continue
            if args.use_clean_text_only and clean_embed is None:
                continue

            label = sample.get("label")
            if torch.is_tensor(label):
                label = int(label.item())
            else:
                label = int(label)

            audio_t = _to_tensor(audio_embed, device)
            text_t = _to_tensor(clean_embed, device) if args.use_clean_text_only else _to_tensor(asr_embed, device)

            with torch.no_grad():
                logits = model(
                    text_asr=text_t,
                    audio=audio_t,
                    mode="student",
                    return_all=False,
                )
            pred = int(torch.argmax(logits, dim=1).item())
            preds.append(pred)
            targets.append(label)

        wa, ua, wf1, mf1 = _compute_metrics(targets, preds)
        metrics.append({"seed": seed, "WA": wa, "UA": ua, "WF1": wf1, "MF1": mf1})
        print(
            f"[Seed {seed}] WA={wa:.4f} UA={ua:.4f} WF1={wf1:.4f} MF1={mf1:.4f} "
            f"(n={len(targets)})"
        )

    if metrics:
        wa_vals = [m["WA"] for m in metrics]
        ua_vals = [m["UA"] for m in metrics]
        wf1_vals = [m["WF1"] for m in metrics]
        mf1_vals = [m["MF1"] for m in metrics]
        wa_mean, wa_std = float(np.mean(wa_vals)), float(np.std(wa_vals, ddof=0))
        ua_mean, ua_std = float(np.mean(ua_vals)), float(np.std(ua_vals, ddof=0))
        wf1_mean, wf1_std = float(np.mean(wf1_vals)), float(np.std(wf1_vals, ddof=0))
        mf1_mean, mf1_std = float(np.mean(mf1_vals)), float(np.std(mf1_vals, ddof=0))
        input_tag = "Clean" if args.use_clean_text_only else "ASR"
        print(
            f"\nMean across seeds ({input_tag} input) -> WA: {wa_mean:.4f}±{wa_std:.4f} | "
            f"UA: {ua_mean:.4f}±{ua_std:.4f} | WF1: {wf1_mean:.4f}±{wf1_std:.4f} | "
            f"MF1: {mf1_mean:.4f}±{mf1_std:.4f}"
        )
        print(f"Test WER (ASR vs Clean): {wer:.4f}")
    else:
        print("No checkpoints available for evaluation.")

    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpt_dir = resolve_checkpoint_dir("saved_model", args.dataset, args.asr_model)
        checkpoint = os.path.join(ckpt_dir, f"AURORA_{args.dataset}_seed{args.seed}.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = AURORA(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    shown = 0
    rows = []
    collect_all = False

    for idx, sample in enumerate(samples):
        clean_text = sample.get("text_raw_dataset")
        asr_text = sample.get("text_raw_asr")
        if not clean_text or not asr_text:
            continue
        if _normalize_text(clean_text) == _normalize_text(asr_text):
            continue

        audio_embed = sample.get("audio_embed")
        asr_embed = sample.get("text_embed_asr")
        clean_embed = sample.get("text_embed_dataset")
        if asr_embed is None:
            asr_embed = sample.get("text_embed")
        if clean_embed is None:
            clean_embed = sample.get("text_embed")
        if audio_embed is None or asr_embed is None or clean_embed is None:
            continue

        label = sample.get("label")
        if torch.is_tensor(label):
            label = int(label.item())
        else:
            label = int(label)

        audio_t = _to_tensor(audio_embed, device)
        asr_t = _to_tensor(asr_embed, device)
        clean_t = _to_tensor(clean_embed, device)
        text_t = clean_t if args.use_clean_text_only else asr_t

        with torch.no_grad():
            logits = model(
                text_asr=text_t,
                audio=audio_t,
                clean_text=clean_t,
                mode="student",
                return_all=False,
            )
        pred = int(torch.argmax(logits, dim=1).item())

        if pred != label:
            continue

        edit_dist, ref_len = _compute_wer(clean_text, asr_text)
        if edit_dist is None or not ref_len:
            continue
        wer = edit_dist / ref_len
        if wer < args.min_wer:
            continue

        sample_id = sample.get("sample_id", f"idx_{idx}")
        row = {
            "sample_id": sample_id,
            "label": label,
            "pred": pred,
            "wer": wer,
            "edit_distance": edit_dist,
            "ref_len": ref_len,
            "clean_text": clean_text,
            "asr_text": asr_text,
        }
        rows.append(row)

        if not collect_all:
            shown += 1
            if args.max_samples and shown >= args.max_samples:
                break

    if args.sort_by_wer and rows:
        rows.sort(key=lambda r: r["wer"], reverse=True)

    if args.save_csv:
        try:
            import pandas as pd
            pd.DataFrame(rows).to_csv(args.save_csv, index=False)
            print(f"\nSaved results to {args.save_csv}")
        except Exception as exc:
            print(f"\nFailed to save CSV: {exc}")

    if not rows:
        print("\nNo samples matched the criteria.")


if __name__ == "__main__":
    main()
