import os
import sys
import torch
import argparse
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.dataloader import get_dataloaders, DATASET_CONFIG
from src.utils.utils import set_seed
from src.training_process.training_stage import train_teacher_student
from src.architecture.configs.asr_models import ASR_MODEL_CONFIGS, DEFAULT_ASR_MODEL_KEY


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--use_clean_text_only", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AURORA")
    parser.add_argument("--wandb_mode", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("saved_model", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset in DATASET_CONFIG:
        args.num_classes = DATASET_CONFIG[args.dataset]["num_classes"]

    seeds = [42,52,103,128,923]
    results = []

    for seed in seeds:
        set_seed(seed)
        loaders = get_dataloaders(args)
        if args.stage1_epochs is None:
            args.stage1_epochs = args.epochs

        run = None
        if args.wandb_mode is not None:
            os.environ["WANDB_MODE"] = args.wandb_mode
        try:
            run = wandb.init(
                project=args.wandb_project,
                name=f"{args.dataset}_seed{seed}_AURORA",
                config={
                    "dataset": args.dataset,
                    "num_classes": args.num_classes,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "stage1_epochs": args.stage1_epochs,
                    "feature_prefix": args.feature_prefix,
                    "asr_model": args.asr_model,
                    "lr": args.lr,
                    "lambda_kd": args.lambda_kd,
                    "temperature": args.temperature,
                    "selection_metric": args.selection_metric,
                    "use_clean_text_only": args.use_clean_text_only,
                    "seed": seed,
                },
            )
        except Exception:
            run = None

        wa, ua, wf1, uf1 = train_teacher_student(args, loaders, device, seed, run)
        results.append({"seed": seed, "WA": wa, "UA": ua, "WF1": wf1,"UF1": uf1})
        if run is not None:
            run.finish()

    if results:
        print("\n=== Final Results ===")
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            print(df)
            print(f"Avg WA: {df['WA'].mean():.4f} +/- {df['WA'].std():.4f}")
            print(f"Avg UA: {df['UA'].mean():.4f} +/- {df['UA'].std():.4f}")
            print(f"Avg WF1: {df['WF1'].mean():.4f} +/- {df['WF1'].std():.4f}")
            print(f"Avg UF1: {df['UF1'].mean():.4f} +/- {df['UF1'].std():.4f}")
        except Exception as exc:
            print(f"Pandas unavailable ({exc}); printing raw results.")
            print(results)
            def _mean_std(values):
                if not values:
                    return 0.0, 0.0
                mean = sum(values) / len(values)
                if len(values) < 2:
                    return mean, 0.0
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                return mean, var ** 0.5
            wa_vals = [r["WA"] for r in results if "WA" in r]
            ua_vals = [r["UA"] for r in results if "UA" in r]
            wf1_vals = [r["WF1"] for r in results if "WF1" in r]
            uf1_vals = [r["UF1"] for r in results if "UF1" in r]
            wa_mean, wa_std = _mean_std(wa_vals)
            ua_mean, ua_std = _mean_std(ua_vals)
            wf1_mean, wf1_std = _mean_std(wf1_vals)
            uf1_mean, uf1_std = _mean_std(uf1_vals)
            print(f"Avg WA: {wa_mean:.4f} +/- {wa_std:.4f}")
            print(f"Avg UA: {ua_mean:.4f} +/- {ua_std:.4f}")
            print(f"Avg WF1: {wf1_mean:.4f} +/- {wf1_std:.4f}")
            print(f"Avg UF1: {uf1_mean:.4f} +/- {uf1_std:.4f}")
    else:
        print("\nTraining not executed; no WA/UA metrics to report.")


if __name__ == "__main__":
    main()
