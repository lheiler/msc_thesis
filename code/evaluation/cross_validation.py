"""
cross_validation.py — 5-fold subject-wise cross-validation with linear probe.

Provides publication-grade evaluation: k-fold CV, logistic regression baseline,
mean ± std metrics, and pairwise statistical tests between methods.

Usage (standalone):
    from evaluation.cross_validation import CrossValidator
    cv = CrossValidator(n_splits=5, n_trials=30)
    results = cv.run(X, y, sample_ids, task_type='classification')

Usage (from main.py with --cv flag):
    Integrated automatically; see main.py.
"""

from __future__ import annotations

import re
import copy
import random
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy import stats as sp_stats

from evaluation.model_training.single_task_model import SingleTaskModel, train as train_single_task
from evaluation.model_training.optuna_search import tune_hyperparameters

__all__ = ["CrossValidator", "LogisticProbe", "StatisticalTests", "cv_results_to_dict"]


# =====================================================================
# Reproducibility
# =====================================================================

_SEED = 42


def _set_seed(seed: int = _SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Subject ID extraction (shared with main.py)
# =====================================================================

def extract_subject_id(sample_id: str) -> str:
    """Extract base subject ID from a sample ID.

    Handles both TUH and BIDS/LEMON formats:
      TUH:   'aaaaapjb_s001_t001_epoch0000' → 'aaaaapjb'
      LEMON: 'sub-032400_EC_epoch0'         → 'sub-032400'
    """
    # BIDS format (LEMON): sub-XXXXXX_...
    m_bids = re.match(r"^(sub-[A-Za-z0-9]+)", sample_id)
    if m_bids:
        return m_bids.group(1)
    # TUH format: subjectid_sXXX_tXXX_epochXXXX
    m_tuh = re.match(r"^([A-Za-z0-9]+)_s\d+", sample_id)
    if m_tuh:
        return m_tuh.group(1)
    # Fallback markers
    for mkr in ["_s", "_t", "_epoch"]:
        if mkr in sample_id:
            return sample_id.split(mkr, 1)[0]
    if "_" in sample_id:
        return sample_id.split("_", 1)[0]
    return sample_id


# =====================================================================
# Logistic / Linear Probe
# =====================================================================

class LogisticProbe:
    """Scikit-learn based linear probe for latent feature evaluation.

    Classification: LogisticRegression (class_weight='balanced')
    Regression:     Ridge (alpha=1.0)
    """

    def __init__(self, task_type: str = "classification", num_classes: int = 1):
        self.task_type = task_type
        self.num_classes = num_classes
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.task_type == "classification":
            if self.num_classes > 2:
                self.model = LogisticRegression(
                    max_iter=2000, class_weight="balanced",
                    solver="lbfgs", random_state=_SEED,
                )
            else:
                self.model = LogisticRegression(
                    max_iter=2000, class_weight="balanced",
                    solver="lbfgs", random_state=_SEED,
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
        if getattr(self, "_single_class", None) is not None:
            return np.full(len(X), self._single_class)
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate and return metrics in the same format as SingleTaskModel.evaluate()."""
        metrics: Dict[str, Any] = {}

        if self.task_type == "classification":
            y_pred = self.predict(X)
            y_prob = None
            if getattr(self, "_single_class", None) is None:
                try:
                    y_prob = self.model.predict_proba(X)
                except AttributeError:
                    pass

            metrics["accuracy"] = float(accuracy_score(y, y_pred))

            avg = "macro" if self.num_classes > 2 else "binary"
            prec, rec, f1, _ = precision_recall_fscore_support(
                y, y_pred, average=avg, zero_division=0
            )
            metrics["precision"] = float(prec)
            metrics["recall"] = float(rec)
            metrics["f1"] = float(f1)

            # ROC-AUC
            if y_prob is not None and len(np.unique(y)) == 2:
                try:
                    if y_prob.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y, y_prob[:, 1]))
                        metrics["pr_auc"] = float(average_precision_score(y, y_prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
                        )
                except Exception:
                    pass
        else:
            y_pred = self.predict(X)
            metrics["mae"] = float(mean_absolute_error(y, y_pred))
            metrics["rmse"] = float(mean_squared_error(y, y_pred) ** 0.5)
            try:
                metrics["r2"] = float(r2_score(y, y_pred))
            except Exception:
                pass

        return metrics


# =====================================================================
# Cross-Validator
# =====================================================================

class CrossValidator:
    """5-fold subject-wise cross-validation with Optuna MLP and linear probe.

    Parameters
    ----------
    n_splits : int
        Number of CV folds (max 5 as per user requirement).
    n_trials : int
        Optuna trials for architecture search (only on fold 0).
    batch_size : int
        Batch size for DataLoaders.
    early_stopping_patience : int
        Early stopping patience for MLP training.
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_trials: int = 30,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        device: str = "cpu",
    ):
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
        sample_ids: List[str],
        task_type: Literal["classification", "regression"] = "classification",
        num_classes: int = 1,
        task_name: str = "task",
        ordinal_sigma: Optional[float] = None,
        results_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run k-fold CV and return aggregated results.

        Returns dict with:
            - Per-fold MLP metrics
            - Per-fold logistic probe metrics
            - Mean ± std for each metric
            - Per-fold predictions (for statistical tests)
        """
        _set_seed()

        # Extract subject groups for GroupKFold
        subject_groups = np.array([extract_subject_id(sid) for sid in sample_ids])
        unique_subjects = np.unique(subject_groups)
        effective_splits = min(self.n_splits, len(unique_subjects))
        if effective_splits < self.n_splits:
            print(f"  ⚠ Only {len(unique_subjects)} unique subjects; "
                  f"reducing to {effective_splits} folds")

        gkf = GroupKFold(n_splits=effective_splits)

        # Storage for per-fold results
        mlp_fold_metrics: List[Dict[str, Any]] = []
        probe_fold_metrics: List[Dict[str, Any]] = []
        mlp_fold_preds: List[np.ndarray] = []
        mlp_fold_trues: List[np.ndarray] = []
        probe_fold_preds: List[np.ndarray] = []

        best_arch_spec: Optional[Dict] = None  # discovered on fold 0
        best_optuna_params: Optional[Dict] = None

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        input_dim = X.shape[1]

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=subject_groups)
        ):
            print(f"\n{'='*50}")
            print(f"  Fold {fold_idx + 1}/{effective_splits}  "
                  f"(train={len(train_idx)}, test={len(test_idx)})")
            print(f"{'='*50}")

            # ---- Normalise using fold-specific train stats ----
            X_train_fold = X_tensor[train_idx]
            X_test_fold = X_tensor[test_idx]
            y_train_fold = y_tensor[train_idx]
            y_test_fold = y_tensor[test_idx]

            x_mean = X_train_fold.mean(dim=0, keepdim=True)
            x_std = X_train_fold.std(dim=0, keepdim=True) + 1e-8
            X_train_norm = (X_train_fold - x_mean) / x_std
            X_test_norm = (X_test_fold - x_mean) / x_std

            # ---- DataLoaders ----
            train_ds = TensorDataset(X_train_norm, y_train_fold)
            test_ds = TensorDataset(X_test_norm, y_test_fold)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

            # ---- Split fold-train into train/val for Optuna / early stopping ----
            fold_subject_groups = subject_groups[train_idx]
            gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=_SEED + fold_idx)
            inner_train_idx, inner_val_idx = next(gss_inner.split(np.arange(len(train_idx)), groups=fold_subject_groups))

            inner_train_loader = DataLoader(
                TensorDataset(X_train_norm[inner_train_idx], y_train_fold[inner_train_idx]),
                batch_size=self.batch_size, shuffle=True,
            )
            inner_val_loader = DataLoader(
                TensorDataset(X_train_norm[inner_val_idx], y_train_fold[inner_val_idx]),
                batch_size=self.batch_size, shuffle=False,
            )

            # ==== MLP (Optuna-tuned) ====
            if fold_idx == 0 or best_arch_spec is None:
                # Full Optuna search on fold 0 to find architecture
                print(f"  🔍 Running Optuna HPO ({self.n_trials} trials)...")
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
                print(f"  ✓ Best architecture: {best_arch_spec['hidden_dims']}, "
                      f"dropout={best_arch_spec['dropout']:.2f}, lr={best_optuna_params.get('lr', 1e-3):.5f}")

            # Re-train the best architecture on full fold-train
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

            # Evaluate MLP on fold-test
            fold_metrics = model.evaluate(
                test_loader, output_type=task_type, device=self.device,
                ordinal_sigma=ordinal_sigma,
            )
            mlp_fold_metrics.append(fold_metrics)

            # Collect per-sample predictions for statistical tests
            model.eval()
            fold_preds_list = []
            fold_trues_list = []
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

            print(f"  MLP fold {fold_idx+1}: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in fold_metrics.items()
                              if isinstance(v, (int, float))))

            # ==== Logistic/Linear Probe ====
            X_train_np = X_train_norm.numpy()
            X_test_np = X_test_norm.numpy()
            y_train_np = y_train_fold.numpy()
            y_test_np = y_test_fold.numpy()

            probe = LogisticProbe(task_type=task_type, num_classes=num_classes)
            probe.fit(X_train_np, y_train_np)
            probe_metrics = probe.evaluate(X_test_np, y_test_np)
            probe_fold_metrics.append(probe_metrics)
            probe_fold_preds.append(probe.predict(X_test_np))

            print(f"  Probe fold {fold_idx+1}: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in probe_metrics.items()
                              if isinstance(v, (int, float))))

        # ==== Aggregate ====
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
        mlp_folds: List[Dict],
        probe_folds: List[Dict],
        mlp_preds: List[np.ndarray],
        mlp_trues: List[np.ndarray],
        probe_preds: List[np.ndarray],
        task_name: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Compute mean ± std across folds for all metrics."""

        def _agg(fold_list: List[Dict], prefix: str) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            # Gather all metric keys
            all_keys = set()
            for fm in fold_list:
                all_keys.update(k for k, v in fm.items() if isinstance(v, (int, float)))
            for key in sorted(all_keys):
                vals = [fm[key] for fm in fold_list if key in fm and isinstance(fm[key], (int, float))]
                if vals:
                    out[f"{prefix}.{key}_mean"] = float(np.mean(vals))
                    out[f"{prefix}.{key}_std"] = float(np.std(vals))
                    out[f"{prefix}.{key}_folds"] = [float(v) for v in vals]
            return out

        results: Dict[str, Any] = {}
        results.update(_agg(mlp_folds, "mlp"))
        results.update(_agg(probe_folds, "linear_probe"))

        # Store raw fold data for downstream statistical tests
        results["_mlp_preds"] = mlp_preds
        results["_mlp_trues"] = mlp_trues
        results["_probe_preds"] = probe_preds

        return results


# =====================================================================
# Statistical Tests
# =====================================================================

class StatisticalTests:
    """Pairwise statistical comparison between two methods across CV folds.

    Usage:
        st = StatisticalTests()
        result = st.compare(
            method_a_folds=[0.78, 0.81, 0.79, 0.80, 0.77],
            method_b_folds=[0.74, 0.72, 0.75, 0.73, 0.71],
            metric_name='accuracy'
        )
    """

    @staticmethod
    def compare(
        method_a_folds: List[float],
        method_b_folds: List[float],
        metric_name: str = "metric",
    ) -> Dict[str, Any]:
        """Run paired t-test and Wilcoxon signed-rank test."""
        a = np.array(method_a_folds)
        b = np.array(method_b_folds)
        diff = a - b

        result: Dict[str, Any] = {
            "metric": metric_name,
            "method_a_mean": float(np.mean(a)),
            "method_a_std": float(np.std(a)),
            "method_b_mean": float(np.mean(b)),
            "method_b_std": float(np.std(b)),
            "mean_diff": float(np.mean(diff)),
        }

        # Cohen's d
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        result["cohens_d"] = float(np.mean(diff) / pooled_std) if pooled_std > 1e-12 else 0.0

        # Paired t-test
        if len(a) >= 2:
            t_stat, p_val = sp_stats.ttest_rel(a, b)
            result["paired_ttest_t"] = float(t_stat)
            result["paired_ttest_p"] = float(p_val)
        else:
            result["paired_ttest_t"] = None
            result["paired_ttest_p"] = None

        # Wilcoxon signed-rank (needs n >= 6 ideally, but works with 5)
        try:
            w_stat, w_p = sp_stats.wilcoxon(a, b, alternative="two-sided")
            result["wilcoxon_stat"] = float(w_stat)
            result["wilcoxon_p"] = float(w_p)
        except Exception:
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"] = None

        return result

    @staticmethod
    def compare_all_pairs(
        method_results: Dict[str, Dict[str, Any]],
        metric_key: str = "mlp.accuracy_folds",
    ) -> List[Dict[str, Any]]:
        """Run pairwise comparisons across all method pairs.

        Args:
            method_results: {method_name: cv_results_dict}
            metric_key: Key into cv_results_dict that holds per-fold values
        """
        methods = sorted(method_results.keys())
        comparisons = []
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


# =====================================================================
# Results formatter
# =====================================================================

def cv_results_to_dict(
    cv_results: Dict[str, Any],
    task_name: str,
) -> Dict[str, Any]:
    """Format CV results for integration into final_results (save_results compatible).

    Returns a dict with keys like:
        'cv.abnormal.mlp.accuracy_mean'
        'cv.abnormal.linear_probe.accuracy_mean'
        etc.
    """
    out: Dict[str, Any] = {}
    prefix = f"cv.{task_name}"
    for key, val in cv_results.items():
        if key.startswith("_"):
            continue  # skip internal per-fold arrays
        out[f"{prefix}.{key}"] = val
    return out
