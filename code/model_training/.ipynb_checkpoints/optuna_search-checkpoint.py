import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Literal
import optuna
import copy
import random
import numpy as np

from model_training.single_task_model import SingleTaskModel, train as train_single_task

__all__ = ["tune_hyperparameters"]

# Reproducibility defaults (kept local to this module)
_SEED = 42

def _set_global_seed(seed: int = _SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _suggest_hidden_dims(trial: optuna.Trial, input_dim: int) -> tuple[int, ...]:
    n_layers = trial.suggest_int("n_layers", 2, 4)
    base     = trial.suggest_int("base_width", 64, 512, step=64)
    return tuple(max(int(base * (0.5 ** i)), 16) for i in range(n_layers))


def tune_hyperparameters(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    input_dim: int,
    output_type: Literal["classification", "regression"],
    n_trials: int = 50,
    device: str = "cpu",
    val_split: float = 0.2,
    early_stopping_patience: int = 10,
    results_dir: str | None = None,  # kept for API compatibility; unused
) -> Dict[str, Any]:
    """Run Optuna Bayesian optimisation for a *single* task.

    Designed for reliable, reproducible selection of a final classifier/regressor.
    Uses a seeded TPE sampler, deterministic cuDNN where applicable, and
    minimises the validation loss for both classification and regression.

    Returns a dict with ``best_params``, the trained ``best_model``, and the ``study`` object.
    """

    # Global determinism for this search
    _set_global_seed()

    # If a dedicated validation loader is provided, do not additionally split the train set
    if val_loader is not None:
        val_split = None  # ensure single validation policy

    best_global_loss: float = float("inf")  # lower is better (validation loss)
    best_state_dict: dict | None = None
    best_arch_spec: dict | None = None

    direction = "minimize"
    sampler = optuna.samplers.TPESampler(seed=_SEED)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial: optuna.Trial):
        # ---------------- Hyper-parameter suggestions -----------------
        lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout      = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        scheduler     = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
        hidden_dims  = _suggest_hidden_dims(trial, input_dim)

        # Per-trial deterministic seed (derived from global seed + trial number)
        trial_seed = _SEED + trial.number
        _set_global_seed(trial_seed)

        # ---------------- Model + training ---------------------------
        model = SingleTaskModel(
            input_dim=input_dim,
            output_type=output_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        info = train_single_task(
            model,
            train_loader,
            val_loader=val_loader,
            n_epochs=100,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            scheduler=scheduler,
            val_split=val_split,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=None,   # no checkpoint files
            plot_dir=None,          # no plot files
        )

        print(f"[Trial {trial.number}] Params: lr={lr:.5f}, dropout={dropout:.2f}, weight_decay={weight_decay:.6f}, scheduler={scheduler}, hidden_dims={hidden_dims}")

        # Use validation loss for optimisation and tracking (lower is better)
        val_loss = info["best_val_metric"]
        objective_value = val_loss

        nonlocal best_global_loss, best_state_dict, best_arch_spec
        if val_loss < best_global_loss:
            best_global_loss = float(val_loss)
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_arch_spec = {
                "input_dim": input_dim,
                "output_type": output_type,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
            }
            print(f"🏅 New global best found (val_loss={val_loss:.4f}) – tracked in memory (no files written)")

        return objective_value

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Rebuild the globally best model purely from in-memory copies
    if best_state_dict is None or best_arch_spec is None:
        raise RuntimeError("Optuna search finished without a valid best model.")

    model_best = SingleTaskModel(
        input_dim=best_arch_spec["input_dim"],
        output_type=best_arch_spec["output_type"],
        hidden_dims=tuple(best_arch_spec["hidden_dims"]),
        dropout=best_arch_spec["dropout"],
    )
    model_best.load_state_dict(copy.deepcopy(best_state_dict))

    print(f"✅ Optuna search finished – best val_loss={best_global_loss:.4f}. Best model kept in memory.")

    return {
        "best_params": {
            "architecture": best_arch_spec,
        },
        "study": study,
        "best_model": model_best,
    }