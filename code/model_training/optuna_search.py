import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Literal
import optuna

from model_training.single_task_model import SingleTaskModel, train as train_single_task

__all__ = ["tune_hyperparameters"]

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
    n_trials: int = 30,
    device: str = "cpu",
    val_split: float = 0.2,
    early_stopping_patience: int = 10,
    results_dir: str | None = None,
) -> Dict[str, Any]:
    """Run Optuna Bayesian optimisation for a *single* task.

    Returns a dict with ``best_params`` and the trained ``best_model``.
    """

    # In-memory trackers only – no file writes
    best_global_score: float = float("-inf")  # higher better (accuracy or -loss)
    best_state_dict: dict | None = None
    best_arch_spec: dict | None = None

    def objective(trial: optuna.Trial):
        # ---------------- Hyper-parameter suggestions -----------------
        lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
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
            checkpoint_path=None,   # no checkpoint files
            plot_dir=None,          # no plot files
        )

        # 👇 Add this line to log current trial parameters
        print(f"[Trial {trial.number}] Params: lr={lr:.5f}, dropout={dropout:.2f}, weight_decay={weight_decay:.6f}, scheduler={scheduler}, hidden_dims={hidden_dims}")
        
        # --------------------------------------------------------------
        # Track *global* best across ALL trials ------------------------
        # --------------------------------------------------------------
        if output_type == "classification":
            current_score = info.get("best_val_score")  # accuracy (higher better)
            objective_value = -current_score  # Optuna minimises → negative accuracy
        else:
            # For regression we minimise val_loss – convert to score by negating
            current_score = -info["best_val_metric"]      # higher is better now
            objective_value = info["best_val_metric"]     # keep loss for optimisation

        nonlocal best_global_score, best_state_dict, best_arch_spec
        if current_score is not None and current_score > best_global_score:
            best_global_score = current_score
            # Capture current best weights and minimal architecture in-memory
            best_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            best_arch_spec = {
                "input_dim": input_dim,
                "output_type": output_type,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
            }
            print(f"🏅 New global best found (score={current_score:.4f}) – tracked in memory (no files written)")

        return objective_value

    study = optuna.create_study(direction="minimize")
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
    model_best.load_state_dict(best_state_dict)

    print(f"✅ Optuna search finished – best score={best_global_score:.4f}. Best model kept in memory (no disk I/O).")

    return {
        "best_params": best_arch_spec,
        "study": study,
        "best_model": model_best,
        # no file path returned – nothing was written
    } 