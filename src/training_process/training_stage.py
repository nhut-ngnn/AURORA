import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.architecture.AURORA import AURORA
from src.training_process.data_io import resolve_checkpoint_dir
from src.utils.utils import compute_metrics, evaluate_stage


def kd_kl_loss(logits_s, logits_t, temperature=2.0):
    log_p_s = F.log_softmax(logits_s / temperature, dim=-1)
    p_t = F.softmax(logits_t / temperature, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature * temperature)


def train_teacher_student(args, loaders, device, seed, run=None):
    print(f"\n[Training] AURORA Teacher-Student - Seed {seed}")

    model = AURORA(num_classes=args.num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss()

    epochs = getattr(args, "stage1_epochs", None)
    if epochs is None:
        epochs = getattr(args, "epochs", 0)

    lambda_kd = getattr(args, "lambda_kd", 1.0)
    temperature = getattr(args, "temperature", 2.0)
    lr = getattr(args, "lr", 1e-4)
    selection_metric = getattr(args, "selection_metric", "val_loss")
    use_clean_text_only = getattr(args, "use_clean_text_only", False)
    input_tag = "Clean" if use_clean_text_only else "ASR"
    lower_is_better = selection_metric == "val_loss"

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler_mode = "min" if lower_is_better else "max"
    scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.3, patience=20)

    best_score = float("inf") if lower_is_better else -float("inf")
    best_epoch = 0
    ckpt_dir = resolve_checkpoint_dir("saved_model", args.dataset, getattr(args, "asr_model", None))
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"AURORA_{args.dataset}_seed{seed}.pt")
    stage1_path = os.path.join(ckpt_dir, f"AURORA_stage1_{args.dataset}_seed{seed}.pt")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"l_total": 0.0, "l_cls": 0.0, "l_kd": 0.0, "l_ce_s": 0.0, "l_ce_t": 0.0}

        for audio, clean_text, asr_text, labels in tqdm(loaders["train"], desc=f"Epoch {epoch + 1}"):
            audio = audio.to(device)
            clean_text = clean_text.to(device)
            asr_text = asr_text.to(device)
            labels = labels.to(device)
            student_text = clean_text if use_clean_text_only else asr_text

            outputs = model(
                text_asr=student_text,
                audio=audio,
                clean_text=clean_text,
                mode="both",
                return_all=True,
            )
            logits_s = outputs["logits_student"]
            logits_t = outputs["logits_teacher"]

            l_ce_s = ce_loss(logits_s, labels)
            l_ce_t = ce_loss(logits_t, labels)
            l_cls = 0.5 * (l_ce_s + l_ce_t)
            l_kd = kd_kl_loss(logits_s, logits_t.detach(), temperature=temperature)
            loss = l_cls + lambda_kd * l_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["l_total"] += float(loss.detach().cpu())
            epoch_losses["l_cls"] += float(l_cls.detach().cpu())
            epoch_losses["l_ce_s"] += float(l_ce_s.detach().cpu())
            epoch_losses["l_ce_t"] += float(l_ce_t.detach().cpu())
            epoch_losses["l_kd"] += float(l_kd.detach().cpu())

        num_batches = len(loaders["train"])
        model.eval()
        val_losses = {k: 0.0 for k in epoch_losses}
        preds, targets = [], []
        with torch.no_grad():
            for audio, clean_text, asr_text, labels in loaders["val"]:
                audio = audio.to(device)
                clean_text = clean_text.to(device)
                asr_text = asr_text.to(device)
                labels = labels.to(device)
                student_text = clean_text if use_clean_text_only else asr_text

                outputs = model(
                    text_asr=student_text,
                    audio=audio,
                    clean_text=clean_text,
                    mode="both",
                    return_all=True,
                )
                logits_s = outputs["logits_student"]
                logits_t = outputs["logits_teacher"]

                l_ce_s = ce_loss(logits_s, labels)
                l_ce_t = ce_loss(logits_t, labels)
                l_cls = 0.5 * (l_ce_s + l_ce_t)
                l_kd = kd_kl_loss(logits_s, logits_t, temperature=temperature)
                l_total = l_cls + lambda_kd * l_kd

                val_losses["l_total"] += float(l_total.detach().cpu())
                val_losses["l_cls"] += float(l_cls.detach().cpu())
                val_losses["l_ce_s"] += float(l_ce_s.detach().cpu())
                val_losses["l_ce_t"] += float(l_ce_t.detach().cpu())
                val_losses["l_kd"] += float(l_kd.detach().cpu())

                preds.append(torch.argmax(logits_s, dim=1).cpu().numpy())
                targets.append(labels.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        val_wa, val_ua, _, _ = compute_metrics(targets, preds)

        train_loss = epoch_losses["l_total"] / max(1, num_batches)
        val_batches = len(loaders["val"])
        val_loss = val_losses["l_total"] / max(1, val_batches)
        selection_values = {
            "val_loss": val_loss,
            "val_wa": val_wa,
            "val_ua": val_ua,
        }
        selection_score = selection_values[selection_metric]
        scheduler.step(selection_score)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val WA ({input_tag}): {val_wa:.4f} | Val UA: {val_ua:.4f} | "
            f"Select {selection_metric}: {selection_score:.4f}"
        )

        if run is not None:
            run.log({
                "train/epoch": epoch + 1,
                "train/l_total": train_loss,
                "train/l_cls": epoch_losses["l_cls"] / max(1, num_batches),
                "train/l_kd": epoch_losses["l_kd"] / max(1, num_batches),
                "train/l_ce_s": epoch_losses["l_ce_s"] / max(1, num_batches),
                "train/l_ce_t": epoch_losses["l_ce_t"] / max(1, num_batches),
                "val/l_total": val_loss,
                "val/WA_ASR": val_wa,
                "val/UA_ASR": val_ua,
                "val/selection_score": selection_score,
            })

        improved = selection_score < best_score if lower_is_better else selection_score > best_score
        if improved:
            best_score = selection_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), stage1_path)
            print(f"Saved model at epoch {best_epoch} with {selection_metric}={best_score:.4f}")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    student_asr_wa, student_asr_ua, student_asr_wf1, student_asr_uf1 = evaluate_stage(
        model,
        loaders["test"],
        device,
        use_clean_text=use_clean_text_only,
        return_all=True,
    )
    print(
        f"Inference ({input_tag} input) -> WA: {student_asr_wa:.4f}, UA: {student_asr_ua:.4f}, "
        f"WF1: {student_asr_wf1:.4f}, UF1: {student_asr_uf1:.4f}"
    )

    if run is not None:
        run.summary["best_model_metric"] = selection_metric
        run.summary["best_model_score"] = best_score
        run.summary["best_model_epoch"] = best_epoch
        run.summary["test_WA_ASR"] = student_asr_wa
        run.summary["test_UA_ASR"] = student_asr_ua
        run.summary["test_WF1_ASR"] = student_asr_wf1
        run.summary["test_UF1_ASR"] = student_asr_uf1

    return student_asr_wa, student_asr_ua, student_asr_wf1, student_asr_uf1
