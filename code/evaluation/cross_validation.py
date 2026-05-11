"""Subject-wise k-fold cross-validation with MLP and linear probe.

Provides publication-grade evaluation: GroupKFold CV, logistic/ridge
baseline, mean +/- std metrics, and pairwise statistical tests.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Any, Literal, Optional

import numpy as np
import torch
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from evaluation.model_training.optuna_search import tune_hyperparameters
from evaluation.model_training.single_task_model import (
    SingleTaskModel,
    train as train_single_task,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CrossValidator",
    "LogisticProbe",
    "StatisticalTests",
    "cv_results_to_dict",
]

_SEED = 42


def _set_seed(seed: int = _SEED) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Subject ID extraction
# ---------------------------------------------------------------------------

def extract_subject_id(sample_id: str) -> str:
    """Extract the base subject identifier from a sample ID.

    Handles TUH (``aaaaapjb_s001_t001_epoch0000``) and
    BIDS/LEMON (``sub-032400_EC_epoch0``) formats.

    Args:
        sample_id: Full sample identifier string.

    Returns:
        Subject-level identifier.
    """
    m_bids = re.match(r"^(sub-[A-Za-z0-9]+)", sample_id)
    if m_bids:
        return m_bids.group(1)
    m_tuh = re.match(r"^([A-Za-z0-9]+)_s\d+", sample_id)
    if m_tuh:
        return m_tuh.group(1)
    for marker in ["_s", "_t", "_epoch"]:
        if marker in sample_id:
            return sample_id.split(marker, 1)[0]
    if "_" in sample_id:
        return sample_id.split("_", 1)[0]
    return sample_id


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LogisticProbe:
    """Scikit-learn linear probe for latent feature evaluation.

    Uses ``LogisticRegression`` for classification and ``Ridge`` for
    regression.

    Args:
        task_type: ``"classification"`` or ``"regression"``.
        num_classes: Number of output classes (1 for binary).
    """

    def __init__(
        self, task_type: str = "classification", num_classes: int = 1
    ) -> None:
        self.task_type = task_type
        self.num_classes = num_classes
        self.model: Any = None
        self._single_class: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticProbe:
        """Fit the probe on training data.

        Args:
            X: Feature matrix ``(n, d)``.
            y: Labels ``(n,)``.

        Returns:
            Self.
        """
        if self.task_type == "classification":
            self.model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=_SEED,
            )
        else:
            self.model = Ridge(alpha=1.0)

        if self.task_type == "classification" and len(np.unique(y)) < 2:
            self._single_class = float(y[0]) if len(y) > 0 else 0.0
        else:
            self._single_class = None
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for *X*."""
        if self._single_class is not None:
            return np.full(len(X), self._single_class)
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Compute evaluation metrics.

        Args:
            X: Feature matrix ``(n, d)``.
            y: Ground-truth labels ``(n,)``.

        Returns:
            Metrics dict compatible with ``SingleTaskModel.evaluate()``.
        """
        metrics: dict[str, Any] = {}

        if self.task_type == "classification":
            y_pred = self.predict(X)
            y_prob = None
            if self._single_class is None:
                try:
                    y_prob = self.model.predict_proba(X)
                except AttributeError:
                    pass

            metrics["accuracy"] = float(accuracy_score(y, y_pred))
            avg = "macro" if self.num_classes > 2 else "binary"
            prec, rec, f1, _ = precision_recall_fscore_support(
                y, y_pred, average=avg, zero_division=0,
            )
            metrics["precision"] = float(prec)
            metrics["recall"] = float(rec)
            metrics["f1"] = float(f1)

            if y_prob is not None and len(np.unique(y)) == 2:
                try:
                    if y_prob.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y, y_prob[:, 1]))
                        metrics["pr_auc"] = float(average_precision_score(y, y_prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
                        )
                except ValueError:
                    pass
        else:
            y_pred = self.predict(X)
            metrics["mae"] = float(mean_absolute_error(y, y_pred))
            metrics["rmse"] = float(mean_squared_error(y, y_pred) ** 0.5)
            try:
                metrics["r2"] = float(r2_score(y, y_pred))
            except ValueError:
                pass

        return metrics


# ---------------------------------------------------------------------------
# Cross-Validator
# ---------------------------------------------------------------------------

class CrossValidator:
    """Subject-wise k-fold cross-validation with Optuna MLP and linear probe.

    Args:
        n_splits: Number of CV folds (2-5).
        n_trials: Optuna trials for architecture search (fold 0 only).
        batch_size: DataLoader batch size.
        early_stopping_patience: Early stopping patience for MLP training.
        device: PyTorch device string.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_trials: int = 30,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        device: str = "cpu",
    ) -> None:
        assert 2 <= n_splits <= 5, "n_splits must be between 2 and 5"
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.patience = early_stopping_patience
        self.device = device

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: list[str],
        task_type: Literal["classification", "regression"] = "classification",
        num_classes: int = 1,
        task_name: str = "task",
        ordinal_sigma: Optional[float] = None,
        results_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run k-fold CV and return aggregated results.

        Args:
            X: Feature matrix ``(n, d)``.
            y: Labels ``(n,)``.
            sample_ids: Per-sample identifiers for subject grouping.
            task_type: ``"classification"`` or ``"regression"``.
            num_classes: Number of output classes.
            task_name: Human-readable task identifier.
            ordinal_sigma: Ordinal regression sigma (if applicable).
            results_dir: Optional directory for saving intermediate results.

        Returns:
            Dict with per-fold metrics, aggregated statistics, and the best
            architecture/hyperparameters discovered during search.
        """
        _set_seed()
        subject_groups = np.array([extract_subject_id(sid) for sid in sample_ids])
        unique_subjects = np.unique(subject_groups)
        effective_splits = min(self.n_splits, len(unique_subjects))
        if effective_splits < self.n_splits:
            logger.warning(
                "Only %d unique subjects; reducing to %d folds",
                len(unique_subjects), effective_splits,
            )

        gkf = GroupKFold(n_splits=effective_splits)

        mlp_fold_metrics: list[dict[str, Any]] = []
        probe_fold_metrics: list[dict[str, Any]] = []
        mlp_fold_preds: list[np.ndarray] = []
        mlp_fold_trues: list[np.ndarray] = []
        probe_fold_preds: list[np.ndarray] = []

        best_arch_spec: Optional[dict] = None
        best_optuna_params: Optional[dict] = None

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        input_dim = X.shape[1]

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=subject_groups)
        ):
            logger.info(
                "Fold %d/%d (train=%d, test=%d)",
                fold_idx + 1, effective_splits, len(train_idx), len(test_idx),
            )

            X_train_fold = X_tensor[train_idx]
            X_test_fold = X_tensor[test_idx]
            y_train_fold = y_tensor[train_idx]
            y_test_fold = y_tensor[test_idx]

            x_mean = X_train_fold.mean(dim=0, keepdim=True)
            x_std = X_train_fold.std(dim=0, keepdim=True) + 1e-8
            X_train_norm = (X_train_fold - x_mean) / x_std
            X_test_norm = (X_test_fold - x_mean) / x_std

            train_ds = TensorDataset(X_train_norm, y_train_fold)
            test_ds = TensorDataset(X_test_norm, y_test_fold)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

            fold_subject_groups = subject_groups[train_idx]
            gss_inner = GroupShuffleSplit(
                n_splits=1, test_size=0.15, random_state=_SEED + fold_idx,
            )
            inner_train_idx, inner_val_idx = next(
                gss_inner.split(np.arange(len(train_idx)), groups=fold_subject_groups)
            )

            inner_train_loader = DataLoader(
                TensorDataset(X_train_norm[inner_train_idx], y_train_fold[inner_train_idx]),
                batch_size=self.batch_size, shuffle=True,
            )
            inner_val_loader = DataLoader(
                TensorDataset(X_train_norm[inner_val_idx], y_train_fold[inner_val_idx]),
                batch_size=self.batch_size, shuffle=False,
            )

            # Optuna HPO on fold 0
            if fold_idx == 0 or best_arch_spec is None:
                logger.info("Running Optuna HPO (%d trials)...", self.n_trials)
                search_out = tune_hyperparameters(
                    inner_train_loader,
                    inner_val_loader,
                    input_dim=input_dim,
                    output_type=task_type,
                    num_classes=num_classes,
                    n_trials=self.n_trials,
                    device=self.device,
                    early_stopping_patience=self.patience,
                    ordinal_sigma=ordinal_sigma,
                )
                best_optuna_params = search_out["best_params"]
                best_arch_spec = best_optuna_params["architecture"]
                logger.info(
                    "Best architecture: %s, dropout=%.2f, lr=%.5f",
                    best_arch_spec["hidden_dims"],
                    best_arch_spec["dropout"],
                    best_optuna_params.get("lr", 1e-3),
                )

            _set_seed(_SEED + fold_idx)
            model = SingleTaskModel(
                input_dim=input_dim,
                output_type=task_type,
                hidden_dims=tuple(best_arch_spec["hidden_dims"]),
                dropout=best_arch_spec["dropout"],
                num_classes=best_arch_spec.get("num_classes", num_classes),
            )

            train_single_task(
                model,
                inner_train_loader,
                val_loader=inner_val_loader,
                n_epochs=300,
                lr=best_optuna_params.get("lr", 1e-3),
                weight_decay=best_optuna_params.get("weight_decay", 1e-4),
                device=self.device,
                scheduler=best_optuna_params.get("scheduler", "plateau"),
                early_stopping_patience=self.patience,
                ordinal_sigma=ordinal_sigma,
            )

            fold_metrics = model.evaluate(
                test_loader, output_type=task_type, device=self.device,
                ordinal_sigma=ordinal_sigma,
            )
            mlp_fold_metrics.append(fold_metrics)

            model.eval()
            fold_preds_list: list[np.ndarray] = []
            fold_trues_list: list[np.ndarray] = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(self.device).float()
                    out = model(xb)
                    if task_type == "classification":
                        if num_classes > 1:
                            preds = torch.argmax(out, dim=-1).float()
                        else:
                            preds = (torch.sigmoid(out) >= 0.5).float()
                    else:
                        preds = out
                    fold_preds_list.append(preds.cpu().numpy())
                    fold_trues_list.append(yb.numpy())
            mlp_fold_preds.append(np.concatenate(fold_preds_list))
            mlp_fold_trues.append(np.concatenate(fold_trues_list))

            logger.info(
                "MLP fold %d: %s", fold_idx + 1,
                ", ".join(
                    f"{k}={v:.4f}" for k, v in fold_metrics.items()
                    if isinstance(v, (int, float))
                ),
            )

            # Linear probe
            probe = LogisticProbe(task_type=task_type, num_classes=num_classes)
            probe.fit(X_train_norm.numpy(), y_train_fold.numpy())
            probe_metrics = probe.evaluate(X_test_norm.numpy(), y_test_fold.numpy())
            probe_fold_metrics.append(probe_metrics)
            probe_fold_preds.append(probe.predict(X_test_norm.numpy()))

            logger.info(
                "Probe fold %d: %s", fold_idx + 1,
                ", ".join(
                    f"{k}={v:.4f}" for k, v in probe_metrics.items()
                    if isinstance(v, (int, float))
                ),
            )

        results = self._aggregate(
            mlp_fold_metrics, probe_fold_metrics,
            mlp_fold_preds, mlp_fold_trues, probe_fold_preds,
            task_name, task_type,
        )
        results["best_architecture"] = best_arch_spec
        results["best_optuna_params"] = best_optuna_params
        return results

    def _aggregate(
        self,
        mlp_folds: list[dict],
        probe_folds: list[dict],
        mlp_preds: list[np.ndarray],
        mlp_trues: list[np.ndarray],
        probe_preds: list[np.ndarray],
        task_name: str,
        task_type: str,
    ) -> dict[str, Any]:
        """Compute mean +/- std across folds for all metrics."""

        def _agg(fold_list: list[dict], prefix: str) -> dict[str, Any]:
            out: dict[str, Any] = {}
            all_keys: set[str] = set()
            for fm in fold_list:
                all_keys.update(
                    k for k, v in fm.items() if isinstance(v, (int, float))
                )
            for key in sorted(all_keys):
                vals = [
                    fm[key] for fm in fold_list
                    if key in fm and isinstance(fm[key], (int, float))
                ]
                if vals:
                    out[f"{prefix}.{key}_mean"] = float(np.mean(vals))
                    out[f"{prefix}.{key}_std"] = float(np.std(vals))
                    out[f"{prefix}.{key}_folds"] = [float(v) for v in vals]
            return out

        results: dict[str, Any] = {}
        results.update(_agg(mlp_folds, "mlp"))
        results.update(_agg(probe_folds, "linear_probe"))
        results["_mlp_preds"] = mlp_preds
        results["_mlp_trues"] = mlp_trues
        results["_probe_preds"] = probe_preds
        return results


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

class StatisticalTests:
    """Pairwise statistical comparison between methods across CV folds."""

    @staticmethod
    def compare(
        method_a_folds: list[float],
        method_b_folds: list[float],
        metric_name: str = "metric",
    ) -> dict[str, Any]:
        """Run paired t-test and Wilcoxon signed-rank test.

        Args:
            method_a_folds: Per-fold metric values for method A.
            method_b_folds: Per-fold metric values for method B.
            metric_name: Name of the metric being compared.

        Returns:
            Dict with test statistics, p-values, and effect size.
        """
        a = np.array(method_a_folds)
        b = np.array(method_b_folds)
        diff = a - b

        result: dict[str, Any] = {
            "metric": metric_name,
            "method_a_mean": float(np.mean(a)),
            "method_a_std": float(np.std(a)),
            "method_b_mean": float(np.mean(b)),
            "method_b_std": float(np.std(b)),
            "mean_diff": float(np.mean(diff)),
        }

        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        result["cohens_d"] = (
            float(np.mean(diff) / pooled_std) if pooled_std > 1e-12 else 0.0
        )

        if len(a) >= 2:
            t_stat, p_val = sp_stats.ttest_rel(a, b)
            result["paired_ttest_t"] = float(t_stat)
            result["paired_ttest_p"] = float(p_val)
        else:
            result["paired_ttest_t"] = None
            result["paired_ttest_p"] = None

        try:
            w_stat, w_p = sp_stats.wilcoxon(a, b, alternative="two-sided")
            result["wilcoxon_stat"] = float(w_stat)
            result["wilcoxon_p"] = float(w_p)
        except ValueError:
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"] = None

        return result

    @staticmethod
    def compare_all_pairs(
        method_results: dict[str, dict[str, Any]],
        metric_key: str = "mlp.accuracy_folds",
    ) -> list[dict[str, Any]]:
        """Run pairwise comparisons across all method pairs.

        Args:
            method_results: ``{method_name: cv_results_dict}``.
            metric_key: Key holding per-fold metric values.

        Returns:
            List of comparison result dicts.
        """
        methods = sorted(method_results.keys())
        comparisons: list[dict[str, Any]] = []
        for i, m_a in enumerate(methods):
            for m_b in methods[i + 1:]:
                folds_a = method_results[m_a].get(metric_key, [])
                folds_b = method_results[m_b].get(metric_key, [])
                if not folds_a or not folds_b:
                    continue
                comp = StatisticalTests.compare(folds_a, folds_b, metric_name=metric_key)
                comp["method_a"] = m_a
                comp["method_b"] = m_b
                comparisons.append(comp)
        return comparisons


# ---------------------------------------------------------------------------
# Results formatter
# ---------------------------------------------------------------------------

def cv_results_to_dict(
    cv_results: dict[str, Any], task_name: str
) -> dict[str, Any]:
    """Format CV results for integration into ``save_results()``.

    Args:
        cv_results: Raw CV results dict.
        task_name: Task identifier for key prefixing.

    Returns:
        Flat dict with keys like ``cv.abnormal.mlp.accuracy_mean``.
    """
    out: dict[str, Any] = {}
    prefix = f"cv.{task_name}"
    for key, val in cv_results.items():
        if key.startswith("_"):
            continue
        out[f"{prefix}.{key}"] = val
    return out
