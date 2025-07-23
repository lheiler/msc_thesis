"""optuna_tune.py – Hyper-parameter search for CwA-T on TUH EEG.

Usage
-----
python optuna_tune.py configs/encoderL+transformer.yml --n_trials 20

This script:
1. Loads the base YAML config.
2. For each Optuna trial, samples a set of hyper-parameters (LR, weight-decay, d_model, n_head, batch_size).
3. Calls the existing `train` function from train_tuh.py which returns best val accuracy and the path to the saved model.
4. Tracks the best model path across all trials and copies it to `<checkpoint_dir>/<model_name>/optuna_best.pth`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import yaml
import optuna

# Make sure we can import train_tuh.py regardless of working directory
from train_tuh import train  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Base YAML config file")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--study_name", default="cwat_tuning", help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL (sqlite, postgres, …)")
    return parser.parse_args()


def build_search_space(cfg: dict, trial: optuna.trial.Trial) -> dict:
    """Clone the cfg dict and apply trial-specific suggestions."""
    cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep-copy via round-trip

    # Optimiser params
    cfg["optimizer"]["init_lr"] = trial.suggest_float("init_lr", 1e-5, 3e-4, log=True)
    cfg["optimizer"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Cosine schedule minimum LR
    cfg["scheduler"]["lr_min"] = trial.suggest_float("lr_min", 1e-6, 1e-4, log=True)

    # Model architecture
    cfg["model"]["d_model"] = trial.suggest_categorical("d_model", [128, 256, 384])
    cfg["model"]["n_head"] = trial.suggest_categorical("n_head", [2, 4, 8])

    # Batch size (note: adjust accum steps externally if memory limited)
    cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])

    return cfg


def main():
    args = parse_args()
    base_cfg_path = Path(args.config_file)
    with base_cfg_path.open("r") as f:
        base_cfg = yaml.safe_load(f)

    # We'll retrain the best set of hyper-parameters after optimisation, so
    # we do NOT need to keep any intermediate model files.

    def objective(trial: optuna.trial.Trial):
        trial_cfg = build_search_space(base_cfg, trial)

        best_acc, _best_path = train(trial_cfg)  # path discarded to save space

        return best_acc

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    # ------------------------------------------------------------------
    #  Retrain once with the BEST hyper-parameters to obtain the final
    #  weights (only one checkpoint file on disk).
    # ------------------------------------------------------------------

    best_trial = study.best_trial

    # Re-apply params to base config
    best_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy
    for key, value in best_trial.params.items():
        if key in {"init_lr", "weight_decay"}:
            best_cfg["optimizer"][key] = value
        elif key == "lr_min":
            best_cfg["scheduler"]["lr_min"] = value
        elif key in {"d_model", "n_head"}:
            best_cfg["model"][key] = value
        elif key == "batch_size":
            best_cfg["train"]["batch_size"] = value

    # Run a final training with the best configuration
    final_acc, final_model_path = train(best_cfg)

    # Persist the best model at a predictable location
    ckpt_root = Path(best_cfg["checkpoint"]["checkpoint_dir"]).expanduser()
    optuna_best_path = ckpt_root / best_cfg["model"]["name"] / "optuna_best.pth"
    optuna_best_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(final_model_path, optuna_best_path)

    print("\n========== Optuna finished ==========")
    print("Best trial:")
    print(f"  Value (accuracy): {best_trial.value:.4f}")
    print("  Params:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    print(f"\nRetrained best model val_acc={final_acc:.4f}")
    print(f"Best model saved to: {optuna_best_path}\n")


if __name__ == "__main__":
    main() 