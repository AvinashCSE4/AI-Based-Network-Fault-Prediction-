"""
train_model.py  -  NetGuard AI : AI-Based Network Fault Prediction
===================================================================
This file contains everything that is NOT user-interface code:

  1. Network parameter definitions and the (configurable) threshold table.
  2. A documented, rule-based fault-severity function used to create labels.
  3. Synthetic data generators (training data + time-series monitoring data).
  4. Model training / comparison (Decision Tree, Logistic Regression, Random Forest).
  5. Prediction + explanation helpers used by the dashboard (app.py).
  6. CSV validation helpers.

You can run this file directly to (re)train the model:

    python train_model.py

IMPORTANT (honest scope note):
  * The training data is SYNTHETIC. The numeric ranges below are demonstration
    assumptions, NOT universal networking standards.
  * Labels come from the rule-based function `rule_based_labels()`. The ML model
    therefore learns to approximate that rule from noisy synthetic samples.
    High accuracy here does not prove accuracy on real production networks.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# 1. Paths and constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = MODEL_DIR / "fault_prediction_model.pkl"
DATASET_PATH = DATA_DIR / "network_dataset.csv"

RANDOM_SEED = 42  # fixed seed -> reproducible data and model

LABELS = ["Normal", "Warning", "Critical"]
RISK_LEVELS = {"Normal": "Low", "Warning": "Medium", "Critical": "High"}

# Parameters used as ML features.
#   direction "high": a HIGH value is bad  (latency, loss, CPU ...)
#   direction "low" : a LOW value is bad   (throughput)
#   warn / crit     : demonstration thresholds (edit them to experiment!)
#   min / max       : allowed range (also used by the Simulation Lab sliders)
FEATURE_INFO = {
    "latency": {"name": "Network latency", "short": "Latency", "unit": "ms", "direction": "high",
                "warn": 100, "crit": 250, "min": 1, "max": 500, "default": 20, "step": 1},
    "packet_loss": {"name": "Packet loss", "short": "Packet Loss", "unit": "%", "direction": "high",
                    "warn": 1, "crit": 5, "min": 0, "max": 100, "default": 0, "step": 0.5},
    "bandwidth_utilization": {"name": "Bandwidth utilization", "short": "Bandwidth", "unit": "%",
                              "direction": "high", "warn": 70, "crit": 90, "min": 0, "max": 100,
                              "default": 30, "step": 1},
    "throughput": {"name": "Throughput", "short": "Throughput", "unit": "Mbps", "direction": "low",
                   "warn": 300, "crit": 100, "min": 0, "max": 1000, "default": 450, "step": 5},
    "cpu_utilization": {"name": "CPU utilization", "short": "CPU", "unit": "%", "direction": "high",
                        "warn": 70, "crit": 90, "min": 0, "max": 100, "default": 35, "step": 1},
    "memory_utilization": {"name": "Memory utilization", "short": "Memory", "unit": "%",
                           "direction": "high", "warn": 75, "crit": 90, "min": 0, "max": 100,
                           "default": 40, "step": 1},
    "network_traffic": {"name": "Network traffic", "short": "Traffic", "unit": "Mbps",
                        "direction": "high", "warn": 700, "crit": 900, "min": 0, "max": 1000,
                        "default": 400, "step": 5},
    "error_rate": {"name": "Error rate", "short": "Error Rate", "unit": "%", "direction": "high",
                   "warn": 1, "crit": 5, "min": 0, "max": 100, "default": 0, "step": 0.5},
}
FEATURES = list(FEATURE_INFO.keys())

# Severity scoring: each feature earns 0 (ok), 1 (warning level) or 2 (critical level) points.
MAX_POINTS = 2 * len(FEATURES)        # 16
WARNING_MIN_POINTS = 3                # total points >= 3  -> Warning
CRITICAL_MIN_POINTS = 8               # total points >= 8  -> Critical

# Typical value distributions (mean, standard deviation) for the synthetic generator.
REGIME_PROFILES = {
    "Normal": {"latency": (25, 10), "packet_loss": (0.2, 0.2), "bandwidth_utilization": (35, 12),
               "throughput": (520, 100), "cpu_utilization": (35, 10), "memory_utilization": (42, 10),
               "network_traffic": (380, 100), "error_rate": (0.2, 0.2)},
    "Warning": {"latency": (110, 35), "packet_loss": (2.5, 1.2), "bandwidth_utilization": (74, 8),
                "throughput": (280, 70), "cpu_utilization": (72, 8), "memory_utilization": (76, 7),
                "network_traffic": (680, 100), "error_rate": (2.2, 1.0)},
    "Critical": {"latency": (300, 80), "packet_loss": (11, 5), "bandwidth_utilization": (94, 4),
                 "throughput": (90, 50), "cpu_utilization": (93, 4), "memory_utilization": (92, 4),
                 "network_traffic": (900, 60), "error_rate": (9, 4)},
}

# Demo devices (name -> type + small offsets so devices behave a little differently).
DEVICE_PROFILES = {
    "Router": {"type": "Router", "offset": {"bandwidth_utilization": 5, "latency": -5}},
    "Switch 1": {"type": "Switch", "offset": {"bandwidth_utilization": 3, "cpu_utilization": -8}},
    "Switch 2": {"type": "Switch", "offset": {"bandwidth_utilization": 2, "cpu_utilization": -8}},
    "Server": {"type": "Server", "offset": {"cpu_utilization": 10, "memory_utilization": 10}},
    "Client Devices": {"type": "Endpoint Group", "offset": {"latency": 10}},
}

# Probability of (Normal, Warning, Critical) regimes for each demo scenario.
SCENARIO_WEIGHTS = {
    "Mixed (realistic)": [0.60, 0.25, 0.15],
    "Mostly healthy": [0.90, 0.08, 0.02],
    "Congested network": [0.30, 0.50, 0.20],
    "Failing network": [0.15, 0.25, 0.60],
}

# Ready-made scenarios for the Simulation Lab (same as the project brief).
SIM_SCENARIOS = {
    "Normal Network": {"latency": 20, "packet_loss": 0, "bandwidth_utilization": 30, "throughput": 450,
                       "cpu_utilization": 35, "memory_utilization": 40, "network_traffic": 400, "error_rate": 0},
    "Warning Network": {"latency": 120, "packet_loss": 2, "bandwidth_utilization": 75, "throughput": 280,
                        "cpu_utilization": 70, "memory_utilization": 75, "network_traffic": 650, "error_rate": 2},
    "Critical Network": {"latency": 350, "packet_loss": 15, "bandwidth_utilization": 98, "throughput": 60,
                         "cpu_utilization": 95, "memory_utilization": 92, "network_traffic": 920, "error_rate": 12},
}

# Column names accepted as a "ground truth" label in uploaded CSV files.
LABEL_COLUMN_CANDIDATES = ["fault_label", "label", "target", "condition"]


def ensure_project_folders() -> None:
    """Create the project folders if they do not exist yet."""
    for folder in (DATA_DIR, MODEL_DIR, ASSETS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 2. Rule-based fault severity (used to create training labels)
# ----------------------------------------------------------------------------
def feature_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Give every parameter 0, 1 or 2 severity points:
      0 = below warning threshold, 1 = at/above warning, 2 = at/above critical.
    For throughput the logic is reversed (LOW throughput is bad).
    """
    points = pd.DataFrame(index=df.index)
    for feature in FEATURES:
        info = FEATURE_INFO[feature]
        values = df[feature].astype(float)
        if info["direction"] == "high":
            points[feature] = (values >= info["warn"]).astype(int) + (values >= info["crit"]).astype(int)
        else:
            points[feature] = (values <= info["warn"]).astype(int) + (values <= info["crit"]).astype(int)
    return points


def label_from_points(total_points) -> np.ndarray:
    """Convert total severity points into Normal / Warning / Critical."""
    total_points = np.asarray(total_points)
    return np.select(
        [total_points >= CRITICAL_MIN_POINTS, total_points >= WARNING_MIN_POINTS],
        ["Critical", "Warning"],
        default="Normal",
    )


def rule_based_labels(df: pd.DataFrame) -> pd.Series:
    """Rule-based label for every row (the 'documented fault severity function')."""
    total = feature_points(df).sum(axis=1)
    return pd.Series(label_from_points(total.values), index=df.index, name="fault_label")


# ----------------------------------------------------------------------------
# 3. Synthetic data generation
# ----------------------------------------------------------------------------
def _clip_to_range(values: dict) -> dict:
    """Keep each generated value inside the allowed range of its parameter."""
    clipped = {}
    for feature, value in values.items():
        info = FEATURE_INFO[feature]
        clipped[feature] = round(float(np.clip(value, info["min"], info["max"])), 2)
    return clipped


def generate_training_data(n_rows: int = 6000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate independent synthetic network samples for model training.
    Roughly 45% healthy-like, 30% degraded-like, 25% failing-like samples, plus
    ~12% 'single spike' samples (one metric is bad but the rest are fine) so the
    model learns that one isolated spike is not automatically a fault.
    The label is ALWAYS computed from the values by `rule_based_labels`.
    """
    rng = np.random.default_rng(seed)
    regimes = rng.choice(LABELS, size=n_rows, p=[0.45, 0.30, 0.25])
    rows = []
    for regime in regimes:
        profile = REGIME_PROFILES[regime]
        row = {f: rng.normal(*profile[f]) for f in FEATURES}
        if regime == "Normal" and rng.random() < 0.25:  # single spike sample
            spike_feature = rng.choice(FEATURES)
            row[spike_feature] = rng.normal(*REGIME_PROFILES["Critical"][spike_feature])
        rows.append(_clip_to_range(row))
    df = pd.DataFrame(rows, columns=FEATURES)
    df["fault_label"] = rule_based_labels(df)
    return df


def generate_monitoring_dataset(seed: int = RANDOM_SEED, steps: int = 120,
                                scenario: str = "Mixed (realistic)",
                                end_time: datetime | None = None,
                                interval_minutes: int = 5) -> pd.DataFrame:
    """
    Generate a SIMULATED time series (one reading every `interval_minutes`) for the
    demo network devices. Each device slowly moves between Normal / Warning /
    Critical behaviour, which makes the charts look like real monitoring data.
    """
    rng = np.random.default_rng(seed)
    weights = SCENARIO_WEIGHTS.get(scenario, SCENARIO_WEIGHTS["Mixed (realistic)"])
    end_time = end_time or datetime.now().replace(second=0, microsecond=0)
    timestamps = [end_time - timedelta(minutes=interval_minutes * (steps - 1 - i)) for i in range(steps)]

    rows = []
    for device, meta in DEVICE_PROFILES.items():
        regime = rng.choice(LABELS, p=weights)
        for stamp in timestamps:
            if rng.random() > 0.78:  # 22% chance to change behaviour at each step
                regime = rng.choice(LABELS, p=weights)
            values = {}
            for feature in FEATURES:
                mean, std = REGIME_PROFILES[regime][feature]
                values[feature] = rng.normal(mean, std) + meta["offset"].get(feature, 0)
            row = {"timestamp": stamp, "device": device, "device_type": meta["type"]}
            row.update(_clip_to_range(values))
            rows.append(row)

    df = pd.DataFrame(rows)
    df["connection_status"] = np.where(df["packet_loss"] >= 95, "Offline", "Online")
    df["fault_label"] = rule_based_labels(df)
    return df


def ensure_sample_dataset() -> Path:
    """Create data/network_dataset.csv (sample dataset) if it is missing."""
    ensure_project_folders()
    if not DATASET_PATH.exists():
        generate_monitoring_dataset(seed=RANDOM_SEED).to_csv(DATASET_PATH, index=False)
    return DATASET_PATH


# ----------------------------------------------------------------------------
# 4. Model training and comparison
# ----------------------------------------------------------------------------
def _build_candidates(seed: int) -> dict:
    """The three algorithms compared in the project."""
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1),
    }


def _feature_importance(model) -> dict | None:
    """Return {feature: importance} for tree-based models, otherwise None."""
    final_step = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(final_step, "feature_importances_"):
        return {f: float(v) for f, v in zip(FEATURES, final_step.feature_importances_)}
    return None


def train_and_evaluate(df: pd.DataFrame, label_col: str = "fault_label", seed: int = RANDOM_SEED,
                       data_source: str = "Synthetic data") -> dict:
    """
    Split the data, train the 3 candidate models, evaluate each one on the TEST set,
    pick the best one and return a 'bundle' dictionary (model + metrics + metadata).

    Selection rule: highest weighted F1-score on the test set. Random Forest is the
    project's primary model, so it wins if it is within 0.005 of the best score.
    """
    X = df[FEATURES].astype(float)
    y = df[label_col].astype(str)
    if y.nunique() < 2:
        raise ValueError("At least two different label classes are needed to train a model.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=stratify)

    labels_present = [label for label in LABELS if label in set(y)]
    results, fitted = [], {}
    for name, model in _build_candidates(seed).items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0)
        results.append({
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-score": float(f1),
        })
        fitted[name] = (model, predictions)

    best_f1 = max(r["F1-score"] for r in results)
    rf_f1 = next(r["F1-score"] for r in results if r["Model"] == "Random Forest")
    best_name = "Random Forest" if rf_f1 >= best_f1 - 0.005 else max(results, key=lambda r: r["F1-score"])["Model"]

    model, predictions = fitted[best_name]
    best = next(r for r in results if r["Model"] == best_name)
    report = classification_report(y_test, predictions, labels=labels_present,
                                   output_dict=True, zero_division=0)
    return {
        "model": model,
        "model_name": best_name,
        "features": FEATURES,
        "labels_present": labels_present,
        "metrics": {k: best[k] for k in ("Accuracy", "Precision", "Recall", "F1-score")},
        "per_class": {label: report[label] for label in labels_present},
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels_present).tolist(),
        "comparison": results,
        "feature_importance": _feature_importance(model),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": seed,
        "data_source": data_source,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_model(bundle: dict, path: Path = MODEL_PATH) -> None:
    ensure_project_folders()
    joblib.dump(bundle, path)


def load_model(path: Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("The saved model file has an unexpected format.")
    return bundle


def train_and_save_default() -> dict:
    """Train on synthetic data, save the model with Joblib, create the sample dataset."""
    ensure_project_folders()
    training_df = generate_training_data()
    bundle = train_and_evaluate(
        training_df, data_source="Synthetic data (6000 generated samples, labels from the rule-based function)")
    save_model(bundle)
    ensure_sample_dataset()
    return bundle


# ----------------------------------------------------------------------------
# 5. Prediction and explanation helpers
# ----------------------------------------------------------------------------
def predict_dataframe(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add AI prediction columns to a DataFrame that contains the 8 feature columns."""
    model = bundle["model"]
    X = df[FEATURES].astype(float)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    out = df.copy()
    best_index = probabilities.argmax(axis=1)
    out["predicted_condition"] = [classes[i] for i in best_index]
    out["confidence"] = probabilities.max(axis=1) * 100
    for label in LABELS:
        out[f"prob_{label.lower()}"] = probabilities[:, classes.index(label)] * 100 if label in classes else 0.0

    points = feature_points(df).sum(axis=1)
    out["severity_points"] = points.values
    out["health_score"] = (100 * (1 - points.values / MAX_POINTS)).round(1)
    out["rule_based_label"] = label_from_points(points.values)
    out["risk_level"] = out["predicted_condition"].map(RISK_LEVELS)
    return out


def explain_record(record: dict) -> list[dict]:
    """
    Explain which input parameters are responsible for a prediction.
    The explanation is computed from the ACTUAL input values and the threshold table.
    Returns a list sorted by severity (most severe first).
    """
    points = feature_points(pd.DataFrame([{f: float(record[f]) for f in FEATURES}])).iloc[0]
    items = []
    for feature in FEATURES:
        info = FEATURE_INFO[feature]
        value = float(record[feature])
        level = int(points[feature])
        if info["direction"] == "high":
            prefix = {0: "Normal", 1: "Elevated", 2: "Very high"}[level]
            rule = f"warning >= {info['warn']:g}, critical >= {info['crit']:g}"
        else:
            prefix = {0: "Normal", 1: "Reduced", 2: "Very low"}[level]
            rule = f"warning <= {info['warn']:g}, critical <= {info['crit']:g}"
        items.append({
            "feature": feature,
            "name": info["name"],
            "value": value,
            "unit": info["unit"],
            "points": level,
            "status": ["Normal", "Warning", "Critical"][level],
            "text": f"{prefix} {info['name'].lower()}: {value:g} {info['unit']} ({rule} {info['unit']})",
        })
    return sorted(items, key=lambda item: item["points"], reverse=True)


def reason_list(record: dict, limit: int = 5) -> list[str]:
    """Short human-readable reasons (only parameters that are above normal)."""
    flagged = [item["text"] for item in explain_record(record) if item["points"] > 0]
    if not flagged:
        return ["All monitored parameters are within the configured normal ranges."]
    return flagged[:limit]


# ----------------------------------------------------------------------------
# 6. CSV reading and validation
# ----------------------------------------------------------------------------
_PARAM_LIMITS = {  # (minimum, maximum) accepted for uploaded values
    "latency": (0, 10000), "packet_loss": (0, 100), "bandwidth_utilization": (0, 100),
    "throughput": (0, 100000), "cpu_utilization": (0, 100), "memory_utilization": (0, 100),
    "network_traffic": (0, 100000), "error_rate": (0, 100),
}
_LABEL_MAP = {"normal": "Normal", "warning": "Warning", "critical": "Critical",
              "fault": "Critical", "critical / fault detected": "Critical"}


def read_uploaded_csv(file_obj) -> pd.DataFrame:
    """Read a CSV file object. Raises ValueError with a friendly message on failure."""
    try:
        return pd.read_csv(file_obj)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty (no data could be read).")
    except UnicodeDecodeError:
        raise ValueError("The file could not be decoded. Please save it as a UTF-8 CSV file.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"The file is not a valid CSV file: {exc}")


def validate_uploaded_dataframe(raw: pd.DataFrame) -> dict:
    """
    Validate an uploaded dataset.
    Returns {"df", "errors", "warnings", "label_col", "labels_valid"}.
    If `errors` is not empty the data must NOT be used.
    """
    result = {"df": None, "errors": [], "warnings": [], "label_col": None, "labels_valid": False}
    errors, warnings = result["errors"], result["warnings"]

    if raw is None or raw.empty or len(raw.columns) == 0:
        errors.append("The uploaded file is empty: it has no data rows.")
        return result

    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    if df.columns.duplicated().any():
        errors.append("The file contains duplicate column names: "
                      + ", ".join(sorted(set(df.columns[df.columns.duplicated()]))))
        return result

    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
        errors.append("Missing required column(s): " + ", ".join(missing_cols)
                      + ". Required columns: " + ", ".join(FEATURES) + ".")
        return result

    numeric = pd.DataFrame(index=df.index)
    for feature in FEATURES:
        original = df[feature]
        converted = pd.to_numeric(original, errors="coerce")
        blank = original.isna() | (original.astype(str).str.strip() == "")
        wrong_type = converted.isna() & ~blank
        if wrong_type.any():
            example = original[wrong_type].iloc[0]
            row_no = int(wrong_type.values.argmax()) + 2  # +2 = header row + 1-based numbering
            errors.append(f"Column '{feature}' has {int(wrong_type.sum())} non-numeric value(s) "
                          f"(for example '{example}' in file row {row_no}). Incorrect data type.")
        if (converted.isna() & blank).any():
            errors.append(f"Column '{feature}' has {int((converted.isna() & blank).sum())} missing value(s). "
                          "Please fill or remove those rows.")
        numeric[feature] = converted
    if errors:
        return result

    for feature in FEATURES:
        low, high = _PARAM_LIMITS[feature]
        column = numeric[feature]
        invalid = ~np.isfinite(column) | (column < low) | (column > high)
        if invalid.any():
            row_no = int(invalid.values.argmax()) + 2
            errors.append(f"Column '{feature}' has {int(invalid.sum())} invalid value(s) outside "
                          f"{low}-{high} (first problem in file row {row_no}, value {column[invalid].iloc[0]}).")
    if errors:
        return result

    for feature in FEATURES:
        df[feature] = numeric[feature].astype(float)

    # Optional columns --------------------------------------------------------
    df["record"] = np.arange(1, len(df) + 1)
    if "device" not in df.columns:
        df["device"] = "Uploaded Records"
        warnings.append("No 'device' column found - all rows are treated as one group.")
    df["device"] = df["device"].fillna("Unknown").astype(str)
    if "device_type" not in df.columns:
        df["device_type"] = "Network Device"
    if "connection_status" not in df.columns:
        df["connection_status"] = np.where(df["packet_loss"] >= 95, "Offline", "Online")
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.isna().any():
            warnings.append("Some 'timestamp' values could not be parsed - record numbers are used on chart axes.")
            df = df.drop(columns=["timestamp"])
        else:
            df["timestamp"] = parsed
            df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Optional ground-truth label column --------------------------------------
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            mapped = df[candidate].astype(str).str.strip().str.lower().map(_LABEL_MAP)
            if mapped.isna().any():
                warnings.append(f"Column '{candidate}' contains labels other than Normal / Warning / Critical, "
                                "so it was ignored for training.")
            else:
                df[candidate] = mapped
                result["label_col"] = candidate
                counts = mapped.value_counts()
                if len(df) >= 50 and len(counts) >= 2 and counts.min() >= 5:
                    result["labels_valid"] = True
                else:
                    warnings.append("Labels found, but training needs >= 50 rows, at least 2 classes and "
                                    ">= 5 rows per class. Using the existing trained model instead.")
            break

    result["df"] = df
    return result


if __name__ == "__main__":
    print("Training NetGuard AI fault prediction model ...")
    trained = train_and_save_default()
    print(f"Selected model : {trained['model_name']}")
    for name, value in trained["metrics"].items():
        print(f"  {name:<10}: {value:.4f}")
    print("\nModel comparison (test set):")
    print(pd.DataFrame(trained["comparison"]).round(4).to_string(index=False))
    print(f"\nModel saved to   : {MODEL_PATH}")
    print(f"Sample dataset   : {DATASET_PATH}")
