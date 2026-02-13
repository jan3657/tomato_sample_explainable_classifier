"""Model constructors used by each fold."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_models_for_fold(fold_index: int, lr_max_iter: int, baseline_strategy: str) -> dict[str, object]:
    """Create per-fold models with deterministic seeds."""
    rf = RandomForestClassifier(random_state=fold_index)
    dt = DecisionTreeClassifier(random_state=fold_index)
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("log_reg", LogisticRegression(max_iter=lr_max_iter, random_state=fold_index)),
        ]
    )
    baseline = DummyClassifier(strategy=baseline_strategy)
    return {"rf": rf, "dt": dt, "lr": lr, "baseline": baseline}
