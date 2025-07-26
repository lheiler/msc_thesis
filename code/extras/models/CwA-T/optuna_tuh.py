#!/usr/bin/env python
"""Optuna hyper-parameter search for CwA-T TUH abnormality training.

This script samples a small set of meaningful hyper-parameters and calls the
``train`` function from ``train_tuh.py`` directly (no subprocess), so it runs in-process
and can share GPU memory.  Val accuracy after training (early-stopped) is the
objective to maximise.  Run overnight:

    python code/extras/models/CwA-T/optuna_tuh.py --config configs/encoderS+transformer.yml \
           --trials 50 --gpu 0
"""

import argparse
import copy
from pathlib import Path
import optuna
import yaml
import multiprocessing
from train_tuh import train  # noqa: after train_tuh edit returns best_acc


SEARCH_SPACE = {
    # Transformer head hyper-params
    "d_model":        [128, 192, 256, 320],      # must be divisible by n_head
    "n_head":         [1, 2, 4, 8],
    "n_layer":        [1, 2, 3],

    # Regularisation / optimiser
    "dropout":        (0.1, 0.6),               # uniform
    "init_lr":        (1e-5, 3e-4),             # loguniform
    "weight_decay":   (1e-4, 3e-2),             # loguniform
    "label_smoothing":(0.0, 0.12),
}


def objective(trial: optuna.Trial, base_cfg: dict) -> float:
    print("[DEBUG] Trial started")
    cfg = copy.deepcopy(base_cfg)

    # Sample hyper-params
    d_model = trial.suggest_categorical("d_model", SEARCH_SPACE["d_model"])
    # pick n_head that divides d_model
    head_options = [h for h in SEARCH_SPACE["n_head"] if d_model % h == 0]
    n_head  = trial.suggest_categorical("n_head", head_options)
    n_layer = trial.suggest_categorical("n_layer", SEARCH_SPACE["n_layer"])

    cfg["model"].update({
        "d_model": d_model,
        "n_head":  n_head,
        "n_layer": n_layer,
        "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
        "classifier": "transformer",
    })
    cfg["optimizer"]["name"] = "adam"  # use Adam
    cfg["optimizer"]["init_lr"]       = trial.suggest_float("init_lr", *SEARCH_SPACE["init_lr"], log=True)
    cfg["optimizer"]["weight_decay"]   = trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"], log=True)
    cfg["criterion"]["label_smoothing"] = trial.suggest_float("label_smoothing", *SEARCH_SPACE["label_smoothing"])

    # Shorten training for each trial (e.g. 25 epochs) but keep early stopping
    cfg["train"]["n_epochs"] = 25
    cfg["train"]["early_stopping"]["patience"] = 5

    best_acc = train(cfg)  # returns best val accuracy
    return -best_acc  # Optuna minimises, so negate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base YAML config file")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["train"]["use_cuda"] = 1
    base_cfg["train"]["gpu_id"] = args.gpu

    print("[DEBUG] About to start optimization")

    def wrapped_objective(trial):
        cfg = copy.deepcopy(base_cfg)

        # Sample hyper-params
        d_model = trial.suggest_categorical("d_model", SEARCH_SPACE["d_model"])
        head_options = [h for h in SEARCH_SPACE["n_head"] if d_model % h == 0]
        n_head  = trial.suggest_categorical("n_head", head_options)
        n_layer = trial.suggest_categorical("n_layer", SEARCH_SPACE["n_layer"])

        cfg["model"].update({
            "d_model": d_model,
            "n_head":  n_head,
            "n_layer": n_layer,
            "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
            "classifier": "transformer",
        })
        cfg["optimizer"]["name"] = "adam"
        cfg["optimizer"]["init_lr"] = trial.suggest_float("init_lr", *SEARCH_SPACE["init_lr"], log=True)
        cfg["optimizer"]["weight_decay"] = trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"], log=True)
        cfg["criterion"]["label_smoothing"] = trial.suggest_float("label_smoothing", *SEARCH_SPACE["label_smoothing"])
        cfg["train"]["n_epochs"] = 25
        cfg["train"]["early_stopping"]["patience"] = 5

        queue = multiprocessing.Queue()

        def worker(cfg, queue):
            from train_tuh import train
            acc = train(cfg)
            queue.put(acc)

        p = multiprocessing.Process(target=worker, args=(cfg, queue))
        p.start()
        p.join()

        if not queue.empty():
            best_acc = queue.get()
            return -best_acc
        else:
            raise RuntimeError("Training process failed or returned no result")

    study = optuna.create_study(direction="minimize")
    study.optimize(wrapped_objective, n_trials=args.trials)

    print("[DEBUG] Optimization finished")

    print("Best trial params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()