import argparse
import copy
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.dataloader import DATASET_CONFIG, get_dataloaders
from src.architecture.AURORA import AURORA
from src.architecture.configs.asr_models import ASR_MODEL_CONFIGS, DEFAULT_ASR_MODEL_KEY
from src.training_process.data_io import resolve_checkpoint_dir


ABLATIONS = {
    "audio_text": {
        "method": "AURORA",
        "modality": "Audio + Text",
        "student_text": "clean",
        "teacher_text": "clean",
        "audio": "real",
        "branch": "student_teacher",
        "use_kd": True,
    },
    "audio_text_transcript": {
        "method": "AURORA",
        "modality": "Audio + Text transcript",
        "student_text": "asr",
        "teacher_text": "clean",
        "audio": "real",
        "branch": "student_teacher",
        "use_kd": True,
    },
    "audio": {
        "method": "AURORA",
        "modality": "Audio",
        "student_text": "zero",
        "teacher_text": "zero",
        "audio": "real",
        "branch": "student_teacher",
        "use_kd": True,
        "eval_from_checkpoint": True,
    },
    "text_transcript": {
        "method": "AURORA",
        "modality": "Text transcript",
        "student_text": "asr",
        "teacher_text": "asr",
        "audio": "zero",
        "branch": "student_teacher",
        "use_kd": True,
        "eval_from_checkpoint": True,
    },
    "no_student_branch": {
        "method": "- w/o Student branch",
        "modality": "Audio + Text transcript",
        "student_text": "asr",
        "teacher_text": "asr",
        "audio": "real",
        "branch": "teacher_only",
        "use_kd": False,
    },
    "no_teacher_branch": {
        "method": "- w/o Teacher branch",
        "modality": "Audio + Text transcript",
        "student_text": "asr",
        "teacher_text": "clean",
        "audio": "real",
        "branch": "student_only",
        "use_kd": False,
    },
    "no_kd": {
        "method": "- w/o L_KD",
        "modality": "Audio + Text transcript",
        "student_text": "asr",
        "teacher_text": "clean",
        "audio": "real",
        "branch": "student_teacher",
        "use_kd": False,
    },
}


def kd_kl_loss(logits_s, logits_t, temperature=2.0):
    log_p_s = F.log_softmax(logits_s / temperature, dim=-1)
    p_t = F.softmax(logits_t / temperature, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature * temperature)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
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

    wa = float((y_true == y_pred).mean())
    ua = float(np.mean(recalls)) if recalls else 0.0
    support_sum = float(np.sum(supports))
    wf1 = float(np.sum(np.array(f1s) * np.array(supports)) / support_sum) if support_sum > 0 else 0.0
    uf1 = float(np.mean(f1s)) if f1s else 0.0
    return wa, ua, wf1, uf1


def _parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_variant_list(value):
    if value == "all":
        return list(ABLATIONS.keys())
    variants = [v.strip() for v in value.split(",") if v.strip()]
    invalid = [v for v in variants if v not in ABLATIONS]
    if invalid:
        raise ValueError("Unknown ablation variant(s): {}".format(", ".join(invalid)))
    return variants


def _select_text(kind, clean_text, asr_text):
    if kind == "clean":
        return clean_text
    if kind == "asr":
        return asr_text
    if kind == "zero":
        return torch.zeros_like(asr_text)
    raise ValueError("Unsupported text input kind: {}".format(kind))


def _select_audio(kind, audio):
    if kind == "real":
        return audio
    if kind == "zero":
        return torch.zeros_like(audio)
    raise ValueError("Unsupported audio input kind: {}".format(kind))


def _forward_for_variant(model, audio, clean_text, asr_text, spec, return_all=True):
    audio_in = _select_audio(spec["audio"], audio)
    student_text = _select_text(spec["student_text"], clean_text, asr_text)
    teacher_text = _select_text(spec["teacher_text"], clean_text, asr_text)

    if spec["branch"] == "teacher_only":
        return model(
            text_asr=teacher_text,
            audio=audio_in,
            clean_text=teacher_text,
            mode="teacher",
            return_all=return_all,
        )

    if spec["branch"] == "student_only":
        return model(
            text_asr=student_text,
            audio=audio_in,
            mode="student",
            return_all=return_all,
        )

    return model(
        text_asr=student_text,
        audio=audio_in,
        clean_text=teacher_text,
        mode="both",
        return_all=return_all,
    )


def _compute_loss(outputs, labels, ce_loss, spec, lambda_kd, temperature):
    if spec["branch"] == "teacher_only":
        logits = outputs["logits_teacher"]
        l_cls = ce_loss(logits, labels)
        zero = torch.zeros((), device=logits.device)
        return l_cls, {
            "l_total": l_cls,
            "l_cls": l_cls,
            "l_ce_s": zero,
            "l_ce_t": l_cls,
            "l_kd": zero,
        }

    if spec["branch"] == "student_only":
        logits = outputs["logits_student"]
        l_cls = ce_loss(logits, labels)
        zero = torch.zeros((), device=logits.device)
        return l_cls, {
            "l_total": l_cls,
            "l_cls": l_cls,
            "l_ce_s": l_cls,
            "l_ce_t": zero,
            "l_kd": zero,
        }

    logits_s = outputs["logits_student"]
    logits_t = outputs["logits_teacher"]
    l_ce_s = ce_loss(logits_s, labels)
    l_ce_t = ce_loss(logits_t, labels)
    l_cls = 0.5 * (l_ce_s + l_ce_t)

    if spec["use_kd"]:
        l_kd = kd_kl_loss(logits_s, logits_t.detach(), temperature=temperature)
    else:
        l_kd = torch.zeros((), device=logits_s.device)

    l_total = l_cls + lambda_kd * l_kd
    return l_total, {
        "l_total": l_total,
        "l_cls": l_cls,
        "l_ce_s": l_ce_s,
        "l_ce_t": l_ce_t,
        "l_kd": l_kd,
    }


def _predict_logits(outputs, spec):
    if spec["branch"] == "teacher_only":
        return outputs["logits_teacher"]
    return outputs["logits_student"]


def evaluate_variant(model, loader, device, spec):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for audio, clean_text, asr_text, labels in loader:
            audio = audio.to(device)
            clean_text = clean_text.to(device)
            asr_text = asr_text.to(device)
            labels = labels.to(device)

            outputs = _forward_for_variant(model, audio, clean_text, asr_text, spec, return_all=True)
            logits = _predict_logits(outputs, spec)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return compute_metrics(targets, preds)


def resolve_pretrained_checkpoint(args, seed):
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.dataset, args.asr_model)
    candidates = [
        os.path.join(ckpt_dir, "AURORA_{}_seed{}.pt".format(args.dataset, seed)),
        os.path.join(ckpt_dir, "AURORA_stage1_{}_seed{}.pt".format(args.dataset, seed)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No pretrained checkpoint found for dataset={} asr_model={} seed={}. Tried: {}".format(
            args.dataset,
            args.asr_model,
            seed,
            ", ".join(candidates),
        )
    )


def evaluate_checkpoint_variant(args, loaders, device, seed, variant_name):
    spec = ABLATIONS[variant_name]
    checkpoint_path = resolve_pretrained_checkpoint(args, seed)
    print(
        "\n[Ablation] {} | {} | {} | seed {} | eval checkpoint".format(
            args.dataset,
            spec["method"],
            spec["modality"],
            seed,
        )
    )
    print("Loaded checkpoint: {}".format(checkpoint_path))

    model = AURORA(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    wa, ua, wf1, uf1 = evaluate_variant(model, loaders["test"], device, spec)
    print("Test | WA={:.4f} UA={:.4f} WF1={:.4f} UF1={:.4f}".format(wa, ua, wf1, uf1))
    return {
        "dataset": args.dataset,
        "variant": variant_name,
        "method": spec["method"],
        "modality": spec["modality"],
        "seed": seed,
        "WA": wa,
        "UA": ua,
        "WF1": wf1,
        "UF1": uf1,
        "best_epoch": "checkpoint",
        "best_score": "",
        "run_type": "eval_checkpoint",
        "checkpoint_path": checkpoint_path,
    }


def train_ablation_variant(args, loaders, device, seed, variant_name):
    spec = ABLATIONS[variant_name]
    print(
        "\n[Ablation] {} | {} | {} | seed {}".format(
            args.dataset,
            spec["method"],
            spec["modality"],
            seed,
        )
    )

    model = AURORA(num_classes=args.num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    lower_is_better = args.selection_metric == "val_loss"
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min" if lower_is_better else "max",
        factor=0.3,
        patience=20,
    )
    best_score = float("inf") if lower_is_better else -float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(args.stage1_epochs):
        model.train()
        train_losses = {"l_total": 0.0, "l_cls": 0.0, "l_ce_s": 0.0, "l_ce_t": 0.0, "l_kd": 0.0}

        for audio, clean_text, asr_text, labels in tqdm(
            loaders["train"],
            desc="{} epoch {}".format(variant_name, epoch + 1),
        ):
            audio = audio.to(device)
            clean_text = clean_text.to(device)
            asr_text = asr_text.to(device)
            labels = labels.to(device)

            outputs = _forward_for_variant(model, audio, clean_text, asr_text, spec, return_all=True)
            loss, parts = _compute_loss(outputs, labels, ce_loss, spec, args.lambda_kd, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key in train_losses:
                train_losses[key] += float(parts[key].detach().cpu())

        val_losses = {"l_total": 0.0, "l_cls": 0.0, "l_ce_s": 0.0, "l_ce_t": 0.0, "l_kd": 0.0}
        model.eval()
        with torch.no_grad():
            for audio, clean_text, asr_text, labels in loaders["val"]:
                audio = audio.to(device)
                clean_text = clean_text.to(device)
                asr_text = asr_text.to(device)
                labels = labels.to(device)

                outputs = _forward_for_variant(model, audio, clean_text, asr_text, spec, return_all=True)
                _, parts = _compute_loss(outputs, labels, ce_loss, spec, args.lambda_kd, args.temperature)
                for key in val_losses:
                    val_losses[key] += float(parts[key].detach().cpu())

        val_wa, val_ua, val_wf1, val_uf1 = evaluate_variant(model, loaders["val"], device, spec)
        num_train_batches = max(1, len(loaders["train"]))
        num_val_batches = max(1, len(loaders["val"]))
        train_loss = train_losses["l_total"] / num_train_batches
        val_loss = val_losses["l_total"] / num_val_batches
        selection_values = {
            "val_loss": val_loss,
            "val_wa": val_wa,
            "val_ua": val_ua,
        }
        selection_score = selection_values[args.selection_metric]
        scheduler.step(selection_score)

        print(
            "Epoch {}/{} | train_loss={:.4f} val_loss={:.4f} val_WA={:.4f} val_UA={:.4f}".format(
                epoch + 1,
                args.stage1_epochs,
                train_loss,
                val_loss,
                val_wa,
                val_ua,
            )
        )

        improved = selection_score < best_score if lower_is_better else selection_score > best_score
        if improved:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    wa, ua, wf1, uf1 = evaluate_variant(model, loaders["test"], device, spec)
    print(
        "Test | best_epoch={} WA={:.4f} UA={:.4f} WF1={:.4f} UF1={:.4f}".format(
            best_epoch,
            wa,
            ua,
            wf1,
            uf1,
        )
    )
    return {
        "dataset": args.dataset,
        "variant": variant_name,
        "method": spec["method"],
        "modality": spec["modality"],
        "seed": seed,
        "WA": wa,
        "UA": ua,
        "WF1": wf1,
        "UF1": uf1,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "run_type": "train",
        "checkpoint_path": "",
    }


def summarize_results(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["variant"], row["method"], row["modality"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dataset, variant, method, modality), items in grouped.items():
        out = {
            "dataset": dataset,
            "variant": variant,
            "method": method,
            "modality": modality,
            "num_seeds": len(items),
            "avg_WA": sum(item["WA"] for item in items) / len(items),
            "avg_UA": sum(item["UA"] for item in items) / len(items),
            "avg_WF1": sum(item["WF1"] for item in items) / len(items),
            "avg_UF1": sum(item["UF1"] for item in items) / len(items),
        }
        for item in items:
            seed = item["seed"]
            out["seed_{}_WA".format(seed)] = item["WA"]
            out["seed_{}_UA".format(seed)] = item["UA"]
            out["seed_{}_WF1".format(seed)] = item["WF1"]
            out["seed_{}_UF1".format(seed)] = item["UF1"]
        summary.append(out)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run AURORA ablations matching the paper table.")
    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--data_dir", type=str, default="features_output")
    parser.add_argument("--feature_prefix", type=str, default=None)
    parser.add_argument(
        "--asr_model",
        type=str,
        default=DEFAULT_ASR_MODEL_KEY,
        choices=sorted(ASR_MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_kd", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--selection_metric", type=str, default="val_loss", choices=["val_loss", "val_wa", "val_ua"])
    parser.add_argument("--checkpoint_dir", type=str, default="saved_model")
    parser.add_argument("--seeds", type=str, default="42,52,103,128,923")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variants or 'all'. Options: {}".format(", ".join(ABLATIONS.keys())),
    )
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--save_raw_csv", type=str, default=None)
    parser.add_argument(
        "--retrain_eval_modalities",
        action="store_true",
        help="Train audio/text_transcript ablations instead of loading the existing AURORA checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset in DATASET_CONFIG:
        args.num_classes = DATASET_CONFIG[args.dataset]["num_classes"]
    if args.stage1_epochs is None:
        args.stage1_epochs = args.epochs

    variants = _parse_variant_list(args.variants)
    seeds = _parse_int_list(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for variant_name in variants:
        for seed in seeds:
            set_seed(seed)
            loaders = get_dataloaders(args)
            spec = ABLATIONS[variant_name]
            if spec.get("eval_from_checkpoint", False) and not args.retrain_eval_modalities:
                rows.append(evaluate_checkpoint_variant(args, loaders, device, seed, variant_name))
            else:
                rows.append(train_ablation_variant(args, loaders, device, seed, variant_name))

    summary = summarize_results(rows)
    print("\n=== Ablation Summary ===")
    for row in summary:
        print(
            "{dataset} | {method} | {modality} | WA={avg_WA:.4f} UA={avg_UA:.4f} "
            "WF1={avg_WF1:.4f} UF1={avg_UF1:.4f}".format(**row)
        )

    default_summary_path = os.path.join("results", "{}_ablation_summary.csv".format(args.dataset))
    default_raw_path = os.path.join("results", "{}_ablation_raw.csv".format(args.dataset))
    write_csv(args.save_csv or default_summary_path, summary)
    write_csv(args.save_raw_csv or default_raw_path, rows)
    print("Saved summary to {}".format(args.save_csv or default_summary_path))
    print("Saved raw seed results to {}".format(args.save_raw_csv or default_raw_path))


if __name__ == "__main__":
    main()
