import os
import json
import shutil
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Literal
import optuna

from model_training.single_task_model import SingleTaskModel, train as train_single_task

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
    results_dir: str | None = None,
) -> Dict[str, Any]:
    """Run Optuna Bayesian optimisation for a *single* task.

    Returns a dict with ``best_params`` and the trained ``best_model``.
    """

    # ------------------------------------------------------------------
    # Prepare results directory & global trackers
    # ------------------------------------------------------------------
    if results_dir is None:
        results_dir = os.getcwd()
    os.makedirs(results_dir, exist_ok=True)

    best_global_score: float = float("-inf")  # higher better (accuracy or -loss)
    best_model_path = os.path.join(results_dir, "optuna_best_model.pth")
    best_arch_path  = os.path.join(results_dir, "optuna_best_arch.json")

    def objective(trial: optuna.Trial):
        # ---------------- Hyper-parameter suggestions -----------------
        lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout      = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        scheduler     = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
        hidden_dims  = _suggest_hidden_dims(trial, input_dim)

        # ---------------- Model + training ---------------------------
        checkpoint_trial_path = os.path.join(results_dir, f"trial_{trial.number}_best.pth")
        model = SingleTaskModel(
            input_dim=input_dim,
            output_type=output_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Each trial gets its own plot sub-folder ----------------------
        plot_dir_trial = os.path.join(results_dir, f"trial_{trial.number}_plots")

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
            checkpoint_path=checkpoint_trial_path,
            plot_dir=plot_dir_trial,
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

        nonlocal best_global_score
        if current_score is not None and current_score > best_global_score:
            best_global_score = current_score

            # ----------------------------------------------------------
            # 1) Persist weights & architecture ------------------------
            # ----------------------------------------------------------
            # Copy checkpoint weights from this trial to global best path
            shutil.copy(checkpoint_trial_path, best_model_path)
            # 2) Copy plots -------------------------------------------
            best_plot_dir = os.path.join(results_dir, "optuna_best_plots")
            # Remove previous best plots dir (if any)
            shutil.rmtree(best_plot_dir, ignore_errors=True)
            try:
                shutil.copytree(plot_dir_trial, best_plot_dir)
            except FileExistsError:
                pass  # shouldn't happen after rmtree

            # Persist minimal architecture spec to rebuild model
            arch_spec = {
                "input_dim": input_dim,
                "output_type": output_type,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
            }
            with open(best_arch_path, "w") as f:
                json.dump(arch_spec, f)

            print(f"🏅 New global best found (score={current_score:.4f}) – model saved to {best_model_path}")

        # Clean up: keep only plots of the best trial ------------------
        if os.path.exists(plot_dir_trial): shutil.rmtree(plot_dir_trial, ignore_errors=True)

        # Remove per-trial checkpoint (always) --------------------------
        try:
            os.remove(checkpoint_trial_path)
        except FileNotFoundError:
            print(f"Checkpoint file not found: {checkpoint_trial_path}")
            pass

        return objective_value

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ------------------------------------------------------------------
    #  Rebuild the globally best model from saved weights --------------
    # ------------------------------------------------------------------
    if not os.path.exists(best_model_path) or not os.path.exists(best_arch_path):
        raise RuntimeError("Best model files were not created during Optuna search.")

    with open(best_arch_path, "r") as f:
        arch_spec = json.load(f)

    model_best = SingleTaskModel(
        input_dim=arch_spec["input_dim"],
        output_type=arch_spec["output_type"],
        hidden_dims=tuple(arch_spec["hidden_dims"]),
        dropout=arch_spec["dropout"],
    )
    model_best.load_state_dict(torch.load(best_model_path, map_location=device))

    print(f"✅ Optuna search finished – best score={best_global_score:.4f}. Model reloaded from disk.")

    return {
        "best_params": arch_spec,
        "study": study,
        "best_model": model_best,
        "best_model_path": best_model_path,
    } 