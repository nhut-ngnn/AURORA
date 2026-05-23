import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.dataloader import DATASET_CONFIG
from src.architecture.AURORA import AURORA
from src.architecture.configs.asr_models import ASR_MODEL_CONFIGS, DEFAULT_ASR_MODEL_KEY
from src.training_process.data_io import build_feature_paths, load_pkl, resolve_checkpoint_dir


def _parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _to_tensor(value, device):
    tensor = torch.as_tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def _wer_components(clean_text, asr_text):
    ref = _normalize_text(clean_text).split()
    hyp = _normalize_text(asr_text).split()
    if not ref:
        return None, None

    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        curr = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1], len(ref)


def _compute_wer(samples):
    edits = 0
    ref_words = 0
    count = 0
    for sample in samples:
        dist, ref_len = _wer_components(sample.get("text_raw_dataset"), sample.get("text_raw_asr"))
        if dist is None or not ref_len:
            continue
        edits += dist
        ref_words += ref_len
        count += 1
    return (edits / ref_words if ref_words else 0.0), count, ref_words


def _compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
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
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    ua = float(np.mean(recalls)) if recalls else 0.0
    support_sum = float(np.sum(supports))
    wf1 = float(np.sum(np.asarray(f1s) * np.asarray(supports)) / support_sum) if support_sum else 0.0
    mf1 = float(np.mean(f1s)) if f1s else 0.0
    return wa, ua, wf1, mf1


def _label_to_int(label):
    if torch.is_tensor(label):
        return int(label.item())
    return int(label)


def _select_text_embedding(sample, mode):
    if mode == "audio_text":
        return sample.get("text_embed_dataset") if sample.get("text_embed_dataset") is not None else sample.get("text_embed")
    if mode == "audio_asr":
        return sample.get("text_embed_asr")
    raise ValueError(f"Unknown evaluation mode: {mode}")


def evaluate_mode(model, samples, device, mode):
    preds = []
    targets = []
    rows = []
    skipped = 0

    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            audio_embed = sample.get("audio_embed")
            text_embed = _select_text_embedding(sample, mode)
            if audio_embed is None or text_embed is None:
                skipped += 1
                continue

            label = _label_to_int(sample.get("label"))
            logits = model(
                text_asr=_to_tensor(text_embed, device),
                audio=_to_tensor(audio_embed, device),
                mode="student",
                return_all=False,
            )
            pred = int(torch.argmax(logits, dim=1).item())
            preds.append(pred)
            targets.append(label)

            rows.append(
                {
                    "idx": idx,
                    "sample_id": sample.get("sample_id", f"idx_{idx}"),
                    "mode": mode,
                    "label": label,
                    "pred": pred,
                    "correct": int(pred == label),
                    "clean_text": sample.get("text_raw_dataset", ""),
                    "asr_text": sample.get("text_raw_asr", ""),
                }
            )

    wa, ua, wf1, mf1 = _compute_metrics(targets, preds)
    return {
        "mode": mode,
        "n": len(targets),
        "skipped": skipped,
        "WA": wa,
        "UA": ua,
        "WF1": wf1,
        "MF1": mf1,
        "rows": rows,
    }


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AURORA on test data with audio+clean text and audio+ASR text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--data_dir", type=str, default="features_output")
    parser.add_argument(
        "--asr_model",
        type=str,
        default=DEFAULT_ASR_MODEL_KEY,
        choices=sorted(ASR_MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="42,52,103,128,923")
    parser.add_argument("--checkpoint", type=str, default=None, help="Evaluate one explicit checkpoint.")
    parser.add_argument("--save_summary_csv", type=str, default=None)
    parser.add_argument("--save_predictions_csv", type=str, default=None)
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
    test_wer, wer_samples, wer_ref_words = _compute_wer(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = _parse_int_list(args.seeds)
    if args.checkpoint is not None:
        seeds = [None]

    summary_rows = []
    prediction_rows = []

    for seed in seeds:
        checkpoint = args.checkpoint
        if checkpoint is None:
            ckpt_dir = resolve_checkpoint_dir("saved_model", args.dataset, args.asr_model)
            checkpoint = os.path.join(ckpt_dir, f"AURORA_{args.dataset}_seed{seed}.pt")
        if not os.path.exists(checkpoint):
            print(f"Checkpoint not found, skipped: {checkpoint}")
            continue

        model = AURORA(num_classes=args.num_classes).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        seed_tag = seed if seed is not None else "checkpoint"
        for mode in ("audio_text", "audio_asr"):
            result = evaluate_mode(model, samples, device, mode)
            row = {
                "seed": seed_tag,
                "mode": mode,
                "asr_model": args.asr_model,
                "checkpoint": checkpoint,
                "n": result["n"],
                "skipped": result["skipped"],
                "WA": result["WA"],
                "UA": result["UA"],
                "WF1": result["WF1"],
                "MF1": result["MF1"],
                "test_WER": test_wer,
                "wer_samples": wer_samples,
                "wer_ref_words": wer_ref_words,
            }
            summary_rows.append(row)
            print(
                "[Seed {}] {} | n={} skipped={} | WA={:.4f} UA={:.4f} WF1={:.4f} MF1={:.4f}".format(
                    seed_tag,
                    mode,
                    result["n"],
                    result["skipped"],
                    result["WA"],
                    result["UA"],
                    result["WF1"],
                    result["MF1"],
                )
            )
            for pred_row in result["rows"]:
                pred_row = {"seed": seed_tag, "asr_model": args.asr_model, **pred_row}
                prediction_rows.append(pred_row)

    print(f"\nTest WER ({args.asr_model} ASR vs clean text): {test_wer:.4f}")

    if summary_rows:
        for mode in ("audio_text", "audio_asr"):
            mode_rows = [r for r in summary_rows if r["mode"] == mode]
            if not mode_rows:
                continue
            print(
                "Mean {} -> WA={:.4f} UA={:.4f} WF1={:.4f} MF1={:.4f}".format(
                    mode,
                    float(np.mean([r["WA"] for r in mode_rows])),
                    float(np.mean([r["UA"] for r in mode_rows])),
                    float(np.mean([r["WF1"] for r in mode_rows])),
                    float(np.mean([r["MF1"] for r in mode_rows])),
                )
            )

    if args.save_summary_csv:
        _write_csv(args.save_summary_csv, summary_rows)
        print(f"Saved summary CSV: {args.save_summary_csv}")
    if args.save_predictions_csv:
        _write_csv(args.save_predictions_csv, prediction_rows)
        print(f"Saved predictions CSV: {args.save_predictions_csv}")


if __name__ == "__main__":
    main()
