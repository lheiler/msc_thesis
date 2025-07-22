import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Literal
import optuna

from model_training.single_task_model import SingleTaskModel, train as train_single_task
from evaluation.single_task_evaluation import evaluate_single_task

__all__ = ["tune_hyperparameters"]

def _suggest_hidden_dims(trial: optuna.Trial, input_dim: int) -> tuple[int, ...]:
    """Sample MLP hidden layer sizes.

    We follow a simple scheme: choose *n_layers* ∈ [1,4] and a *base* hidden
    width.  Widths then decay by ×0.5 each subsequent layer.
    """
    n_layers = trial.suggest_int("n_layers", 2, 4)
    base     = trial.suggest_int("base_width", 64, 512, step=64)
    return tuple(max(int(base * (0.5 ** i)), 16) for i in range(n_layers))


def tune_hyperparameters(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    input_dim: int,
    output_type: Literal["classification", "regression"],
    n_trials: int = 30,
    device: str = "cpu",
    val_split: float = 0.2,
    early_stopping_patience: int = 10,
) -> Dict[str, Any]:
    """Run Optuna Bayesian optimisation for a *single* task.

    Returns a dict with ``best_params`` and the trained ``best_model``.
    """

    def objective(trial: optuna.Trial):
        # ---------------- Hyper-parameter suggestions -----------------
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout      = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        scheduler     = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
        hidden_dims  = _suggest_hidden_dims(trial, input_dim)

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
        )

        # 👇 Add this line to log current trial parameters
        print(f"[Trial {trial.number}] Params: lr={lr:.5f}, dropout={dropout:.2f}, weight_decay={weight_decay:.6f}, scheduler={scheduler}, hidden_dims={hidden_dims}")
        
        # We aim to *minimise* validation loss (best_val_metric from train())
        if output_type == "classification":
            # maximise accuracy → minimise negative accuracy
            val_acc = info.get("best_val_score")
            if val_acc is None:
                raise RuntimeError("Validation accuracy missing from training info.")
            return -val_acc
        else:
            # regression → minimise validation loss (MSE)
            return info["best_val_metric"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params

    # -------- Train final model with the best parameters -------------
    # Re-initialise model with optimal hp and train once more (weights restored)
    hidden_dims_best = _suggest_hidden_dims(study.best_trial, input_dim)
    model_best = SingleTaskModel(
        input_dim=input_dim,
        output_type=output_type,
        hidden_dims=hidden_dims_best,
        dropout=best_params["dropout"],
    )
    train_single_task(
        model_best,
        train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        device=device,
        scheduler=best_params["scheduler"],
        val_split=val_split,
        early_stopping_patience=early_stopping_patience,
    )

    return {
        "best_params": best_params,
        "study": study,
        "best_model": model_best,
    } 