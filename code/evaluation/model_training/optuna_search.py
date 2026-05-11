"""Optuna-based hyperparameter search for single-task models."""
from __future__ import annotations

import copy
import logging
import random
from typing import Any, Literal, Optional

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from evaluation.model_training.single_task_model import (
    SingleTaskModel,
    train as train_single_task,
)

logger = logging.getLogger(__name__)

__all__ = ["tune_hyperparameters"]

_SEED = 42
MAX_TRAIN_EPOCHS = 300


def _set_global_seed(seed: int = _SEED) -> None:
    """Set global random seeds for deterministic Optuna trials."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _suggest_hidden_dims(trial: optuna.Trial, input_dim: int) -> tuple[int, ...]:
    """Suggest a tapered hidden-layer architecture."""
    n_layers = trial.suggest_int("n_layers", 2, 4)
    base = trial.suggest_int(
        "base_width", max(32, input_dim), min(input_dim * 8, 512), step=32,
    )
    return tuple(max(int(base * (0.5 ** i)), 16) for i in range(n_layers))


def tune_hyperparameters(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    input_dim: int,
    output_type: Literal["classification", "regression"],
    num_classes: int = 1,
    n_trials: int = 50,
    device: str = "cpu",
    val_split: Optional[float] = 0.2,
    early_stopping_patience: int = 10,
    results_dir: Optional[str] = None,
    ordinal_sigma: Optional[float] = None,
) -> dict[str, Any]:
    """Run Optuna Bayesian optimisation for a single task.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        input_dim: Feature dimensionality.
        output_type: ``"classification"`` or ``"regression"``.
        num_classes: Number of output classes (1 for binary).
        n_trials: Number of Optuna trials.
        device: PyTorch device string.
        val_split: Validation split ratio (unused when *val_loader* is given).
        early_stopping_patience: Early stopping patience.
        results_dir: Optional directory (currently unused).
        ordinal_sigma: Ordinal regression sigma (if applicable).

    Returns:
        Dict with ``best_params``, ``best_model``, and ``study``.
    """
    _set_global_seed()

    if val_loader is not None:
        val_split = None

    best_global_loss: float = float("inf")
    best_state_dict: Optional[dict] = None
    best_arch_spec: Optional[dict] = None

    sampler = optuna.samplers.TPESampler(seed=_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_global_loss, best_state_dict, best_arch_spec

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        scheduler = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
        hidden_dims = _suggest_hidden_dims(trial, input_dim)

        _set_global_seed(_SEED + trial.number)

        model = SingleTaskModel(
            input_dim=input_dim,
            output_type=output_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
        )

        info = train_single_task(
            model,
            train_loader,
            val_loader=val_loader,
            n_epochs=MAX_TRAIN_EPOCHS,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            scheduler=scheduler,
            val_split=val_split,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=None,
            plot_dir=None,
            ordinal_sigma=ordinal_sigma,
        )

        logger.debug(
            "Trial %d: lr=%.5f dropout=%.2f wd=%.6f sched=%s dims=%s",
            trial.number, lr, dropout, weight_decay, scheduler, hidden_dims,
        )

        val_loss = info["best_val_metric"]
        if val_loss < best_global_loss:
            best_global_loss = float(val_loss)
            best_state_dict = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }
            best_arch_spec = {
                "input_dim": input_dim,
                "output_type": output_type,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "num_classes": num_classes,
            }
            logger.info(
                "New best found (val_loss=%.4f) at trial %d",
                val_loss, trial.number,
            )

        return val_loss

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    if best_state_dict is None or best_arch_spec is None:
        raise RuntimeError("Optuna search finished without a valid best model.")

    model_best = SingleTaskModel(
        input_dim=best_arch_spec["input_dim"],
        output_type=best_arch_spec["output_type"],
        hidden_dims=tuple(best_arch_spec["hidden_dims"]),
        dropout=best_arch_spec["dropout"],
        num_classes=best_arch_spec.get("num_classes", 1),
    )
    model_best.load_state_dict(copy.deepcopy(best_state_dict))

    logger.info("Optuna search finished - best val_loss=%.4f", best_global_loss)

    return {
        "best_params": {
            "architecture": best_arch_spec,
            **study.best_trial.params,
        },
        "study": study,
        "best_model": model_best,
    }
