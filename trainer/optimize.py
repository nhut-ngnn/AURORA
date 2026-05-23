import argparse
import itertools
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.dataloader import DATASET_CONFIG, get_dataloaders
from src.architecture.configs.asr_models import ASR_MODEL_CONFIGS, DEFAULT_ASR_MODEL_KEY
from src.training_process.training_stage import train_teacher_student
from src.utils.utils import set_seed


def _parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search for teacher-student training hyperparameters.")
    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--data_dir", type=str, default="features_output")
    parser.add_argument("--feature_prefix", type=str, default=None)
    parser.add_argument(
        "--asr_model",
        type=str,
        default=DEFAULT_ASR_MODEL_KEY,
        choices=sorted(ASR_MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--asr_models",
        type=str,
        default=None,
        help="Comma-separated ASR model keys. Defaults to --asr_model when omitted.",
    )
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_values", type=str, default="5e-5,1e-4,2e-4")
    parser.add_argument("--lambda_kd_values", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--temperature_values", type=str, default="1.5,2.0,2.5")
    parser.add_argument("--selection_metric", type=str, default="val_loss", choices=["val_loss", "val_wa", "val_ua"])
    parser.add_argument("--use_clean_text_only", action="store_true")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--metric", type=str, default="WA", choices=["WA", "UA", "WF1", "UF1"])
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--save_csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset in DATASET_CONFIG:
        args.num_classes = DATASET_CONFIG[args.dataset]["num_classes"]

    if args.stage1_epochs is None:
        args.stage1_epochs = args.epochs

    lambda_kd_values = _parse_float_list(args.lambda_kd_values)
    temperature_values = _parse_float_list(args.temperature_values)
    lr_values = _parse_float_list(args.lr_values)
    asr_models = [v.strip() for v in args.asr_models.split(",") if v.strip()] if args.asr_models else [args.asr_model]
    invalid_asr_models = [model_key for model_key in asr_models if model_key not in ASR_MODEL_CONFIGS]
    if invalid_asr_models:
        raise ValueError("Unknown ASR model key(s): {}".format(", ".join(invalid_asr_models)))
    seeds = _parse_int_list(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    best = None
    run_count = 0

    for asr_model, lr, lambda_kd, temperature in itertools.product(
        asr_models,
        lr_values,
        lambda_kd_values,
        temperature_values,
    ):
        run_count += 1
        if args.max_runs is not None and run_count > args.max_runs:
            break

        args.asr_model = asr_model
        args.lr = lr
        args.lambda_kd = lambda_kd
        args.temperature = temperature

        metrics = []
        per_seed = {}
        for seed in seeds:
            set_seed(seed)
            loaders = get_dataloaders(args)
            wa, ua, wf1, uf1 = train_teacher_student(args, loaders, device, seed, run=None)
            metrics.append((wa, ua, wf1, uf1))
            per_seed[seed] = {"WA": wa, "UA": ua, "WF1": wf1, "UF1": uf1}

        avg_wa = sum(m[0] for m in metrics) / max(1, len(metrics))
        avg_ua = sum(m[1] for m in metrics) / max(1, len(metrics))
        avg_wf1 = sum(m[2] for m in metrics) / max(1, len(metrics))
        avg_uf1 = sum(m[3] for m in metrics) / max(1, len(metrics))
        score_by_metric = {
            "WA": avg_wa,
            "UA": avg_ua,
            "WF1": avg_wf1,
            "UF1": avg_uf1,
        }
        score = score_by_metric[args.metric]

        result = {
            "asr_model": args.asr_model,
            "lr": lr,
            "lambda_kd": lambda_kd,
            "temperature": args.temperature,
            "selection_metric": args.selection_metric,
            "stage1_epochs": args.stage1_epochs,
            "feature_prefix": args.feature_prefix,
            "avg_WA": avg_wa,
            "avg_UA": avg_ua,
            "avg_WF1": avg_wf1,
            "avg_UF1": avg_uf1,
        }
        for seed in seeds:
            seed_metrics = per_seed.get(seed, {})
            result[f"seed_{seed}_WA"] = seed_metrics.get("WA", None)
            result[f"seed_{seed}_UA"] = seed_metrics.get("UA", None)
            result[f"seed_{seed}_WF1"] = seed_metrics.get("WF1", None)
            result[f"seed_{seed}_UF1"] = seed_metrics.get("UF1", None)
        results.append(result)

        if best is None or score > best["score"]:
            best = {"score": score, **result}

        print(
            "asr={} lr={:.1e} lambda_kd={:.3f} temp={:.2f} select={} | avg_WA={:.4f} avg_UA={:.4f} "
            "avg_WF1={:.4f} avg_UF1={:.4f}".format(
                asr_model,
                lr,
                lambda_kd,
                temperature,
                args.selection_metric,
                avg_wa,
                avg_ua,
                avg_wf1,
                avg_uf1,
            )
        )

    if best is not None:
        print("\nBest (by {}):".format(args.metric))
        print(best)

    if args.save_csv:
        try:
            import pandas as pd

            pd.DataFrame(results).to_csv(args.save_csv, index=False)
            print("Saved results to {}".format(args.save_csv))
        except Exception as exc:
            print("Failed to save CSV: {}".format(exc))


if __name__ == "__main__":
    main()
