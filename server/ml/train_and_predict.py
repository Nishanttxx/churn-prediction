from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_churn_dataset-testing-master.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

NUMERIC = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
CATEGORICAL = ["Gender", "Subscription Type", "Contract Length"]
FEATURES = NUMERIC + CATEGORICAL
EXPECTED = ["CustomerID", *FEATURES, "Churn"]


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def audit(df: pd.DataFrame) -> dict:
    return {
        "source_file": DATA_PATH.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "numeric_columns": NUMERIC,
        "categorical_columns": CATEGORICAL,
        "categorical_unique_values": {k: sorted(df[k].astype(str).unique().tolist()) for k in CATEGORICAL},
        "target": {"name": "Churn", "unique_values": sorted(df["Churn"].unique().tolist()), "counts": {str(k): int(v) for k, v in df["Churn"].value_counts().sort_index().items()}, "interpretation": "1 = churned, 0 = retained"},
        "numeric_ranges": {k: {"min": float(df[k].min()), "max": float(df[k].max()), "mean": round(float(df[k].mean()), 3)} for k in NUMERIC},
        "negative_counts": {k: int((df[k] < 0).sum()) for k in NUMERIC},
        "feature_set": FEATURES,
        "excluded_features": ["CustomerID", "Churn"],
        "cleaning_assumptions": [
            "No rows were removed: the source contains no missing values, duplicate rows, or negative numeric observations.",
            "Churn is already numeric binary 0/1 and is used only as the target.",
            "CustomerID is retained for display only and excluded from preprocessing and model training.",
        ],
    }


def make_preprocessor(scale: bool) -> ColumnTransformer:
    transformers = []
    if scale:
        transformers.append(("num", StandardScaler(), NUMERIC))
    else:
        transformers.append(("num", "passthrough", NUMERIC))
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def models() -> dict[str, tuple[object, bool]]:
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, solver="liblinear", random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=160, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42), False),
        "Support Vector Machine": (CalibratedClassifierCV(LinearSVC(random_state=42, dual="auto", max_iter=2000), method="sigmoid", cv=3), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1), True),
    }


def metrics_for(model, x_test, y_test) -> tuple[dict, np.ndarray, np.ndarray]:
    predicted = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 6),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_test, predicted, zero_division=0)), 6),
    }, predicted, confusion_matrix(y_test, predicted, labels=[0, 1])


def train() -> dict:
    df = load_frame()
    y = df["Churn"].astype(int)
    x = df[FEATURES].copy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    results = {}
    fitted = {}
    started = time.perf_counter()
    for name, (estimator, scale) in models().items():
        model_started = time.perf_counter()
        pipeline = Pipeline([("preprocessor", make_preprocessor(scale)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        metrics, predicted, matrix = metrics_for(pipeline, x_test, y_test)
        results[name] = {**metrics, "training_time_seconds": round(time.perf_counter() - model_started, 3), "confusion_matrix": {"true_negative": int(matrix[0, 0]), "false_positive": int(matrix[0, 1]), "false_negative": int(matrix[1, 0]), "true_positive": int(matrix[1, 1])}}
        fitted[name] = pipeline
    best_name = max(results, key=lambda name: (results[name]["f1_score"], results[name]["recall"], results[name]["precision"]))
    best_pipeline = fitted[best_name]
    pre = best_pipeline.named_steps["preprocessor"]
    feature_names = list(pre.get_feature_names_out())
    rf_pipeline = fitted["Random Forest"]
    rf_pre = rf_pipeline.named_steps["preprocessor"]
    rf_names = list(rf_pre.get_feature_names_out())
    rf_importance = rf_pipeline.named_steps["model"].feature_importances_
    importance = [{"feature": feature, "importance": round(float(value), 8)} for feature, value in sorted(zip(rf_names, rf_importance), key=lambda item: item[1], reverse=True)]
    metadata = {
        "model_name": best_name,
        "metrics": results[best_name],
        "models": results,
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "encoded_feature_count": len(feature_names),
        "selection_rule": "Highest held-out F1 score, then recall, then precision.",
        "split": {"test_size": 0.20, "random_state": 42, "stratified": True},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_training_time_seconds": round(time.perf_counter() - started, 3),
        "dataset_audit": audit(df),
    }
    joblib.dump(best_pipeline, ARTIFACTS / "best_model.joblib", compress=3)
    joblib.dump(fitted, ARTIFACTS / "all_models.joblib", compress=3)
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (ARTIFACTS / "feature_importance.json").write_text(json.dumps(importance, indent=2))
    build_analytics(df, metadata, best_pipeline)
    return metadata


def build_analytics(df: pd.DataFrame, metadata: dict, best_pipeline) -> None:
    churn_rate = df.groupby("Churn").size().to_dict()
    def rate_by(column: str):
        grouped = df.groupby(column)["Churn"].agg(["mean", "count"]).reset_index()
        return [{"name": str(row[column]), "churn_rate": round(float(row["mean"]), 6), "customers": int(row["count"])} for _, row in grouped.iterrows()]
    behavior = []
    for column in NUMERIC:
        grouped = df.groupby(pd.qcut(df[column], q=5, duplicates="drop"))["Churn"].agg(["mean", "count"]).reset_index()
        behavior.append({"feature": column, "buckets": [{"name": str(row[column]), "churn_rate": round(float(row["mean"]), 6), "customers": int(row["count"])} for _, row in grouped.iterrows()]})
    correlation = df[NUMERIC + ["Churn"]].corr().round(4)
    corr_rows = [{"feature": col, **{row: float(correlation.loc[col, row]) for row in correlation.columns}} for col in correlation.columns]
    scored = best_pipeline.predict_proba(df[FEATURES])[:, 1]
    risk_distribution = [
        {"name": "Low risk", "value": int((scored < 0.4).sum())},
        {"name": "Medium risk", "value": int(((scored >= 0.4) & (scored < 0.7)).sum())},
        {"name": "High risk", "value": int((scored >= 0.7).sum())},
    ]
    age_bins = pd.cut(df["Age"], bins=[17, 29, 39, 49, 65], labels=["18–29", "30–39", "40–49", "50–65"])
    demographics = {
        "age": [{"name": str(name), "churn_rate": round(float(group["Churn"].mean()), 6), "customers": int(len(group))} for name, group in df.assign(age_band=age_bins).groupby("age_band", observed=False)],
        "gender": rate_by("Gender"),
    }
    means = df.groupby("Churn")[NUMERIC].mean()
    signals = []
    for feature in NUMERIC:
        delta = float(means.loc[1, feature] - means.loc[0, feature])
        signals.append({"feature": feature, "churned_mean": round(float(means.loc[1, feature]), 3), "retained_mean": round(float(means.loc[0, feature]), 3), "delta": round(delta, 3)})
    analytics = {
        "kpis": {"total_customers": int(len(df)), "churned_customers": int(churn_rate.get(1, 0)), "retained_customers": int(churn_rate.get(0, 0)), "churn_rate": round(float(df["Churn"].mean()), 6)},
        "distribution": [{"name": "Retained", "value": int(churn_rate.get(0, 0))}, {"name": "Churned", "value": int(churn_rate.get(1, 0))}],
        "risk_distribution": risk_distribution,
        "demographics": demographics,
        "by_subscription": rate_by("Subscription Type"),
        "by_contract": rate_by("Contract Length"),
        "behavior": behavior,
        "correlation": corr_rows,
        "behavior_signals": sorted(signals, key=lambda item: abs(item["delta"]), reverse=True),
        "audit": audit(df),
        "feature_importance": json.loads((ARTIFACTS / "feature_importance.json").read_text()),
        "model": metadata,
    }
    (ARTIFACTS / "analytics.json").write_text(json.dumps(analytics, indent=2))


def explore(payload: dict) -> dict:
    df = load_frame()
    page = max(1, int(payload.get("page", 1)))
    page_size = min(50, max(5, int(payload.get("page_size", 10))))
    search = str(payload.get("search", "")).strip().lower()
    if search:
        mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(search, regex=False)).any(axis=1)
        df = df[mask]
    sort_by = payload.get("sort_by")
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=payload.get("sort_dir", "asc") != "desc")
    total = len(df)
    start = (page - 1) * page_size
    rows = df.iloc[start:start + page_size].to_dict(orient="records")
    return {"rows": rows, "total": int(total), "page": page, "page_size": page_size, "pages": int(np.ceil(total / page_size)) if total else 0, "columns": list(load_frame().columns), "audit": audit(load_frame())}


def predict(payload: dict) -> dict:
    model_path = ARTIFACTS / "best_model.joblib"
    if not model_path.exists():
        train()
    model = joblib.load(model_path)
    row = {feature: payload[feature] for feature in FEATURES}
    frame = pd.DataFrame([row], columns=FEATURES)
    probability = float(model.predict_proba(frame)[0][1])
    prediction = int(model.predict(frame)[0])
    risk = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"
    source_frame = load_frame()
    baseline = {feature: float(source_frame[feature].median()) if feature in NUMERIC else str(source_frame[feature].mode().iloc[0]) for feature in FEATURES}
    baseline_frame = pd.DataFrame([baseline], columns=FEATURES)
    baseline_probability = float(model.predict_proba(baseline_frame)[0][1])
    local_effects = []
    for feature in FEATURES:
        counterfactual = frame.copy()
        counterfactual[feature] = baseline[feature]
        counterfactual_probability = float(model.predict_proba(counterfactual)[0][1])
        effect = probability - counterfactual_probability
        local_effects.append({"feature": feature, "effect": round(effect, 6), "absolute_effect": abs(effect)})
    drivers = []
    for item in sorted(local_effects, key=lambda entry: entry["absolute_effect"], reverse=True)[:3]:
        direction = "increases" if item["effect"] > 0 else "reduces"
        drivers.append({"label": f"{item['feature']} {direction} model risk", "detail": f"Changing this input to the source-data median or mode moves the model probability by {abs(item['effect']) * 100:.1f} percentage points."})
    if not drivers: drivers.append({"label": "Composite model signal", "detail": "The risk score reflects the fitted model across all supplied attributes."})
    priority = "Immediate outreach" if risk == "High" else "Proactive check-in" if risk == "Medium" else "Monitor and nurture"
    return {"probability": round(probability, 6), "prediction": prediction, "risk": risk, "priority": priority, "drivers": drivers, "baseline_probability": round(baseline_probability, 6), "model_name": json.loads((ARTIFACTS / "model_metadata.json").read_text())["model_name"]}


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    if command == "train":
        print(json.dumps(train()))
    elif command == "analytics":
        metadata_path = ARTIFACTS / "model_metadata.json"
        if not metadata_path.exists(): train()
        print((ARTIFACTS / "analytics.json").read_text())
    elif command == "audit":
        print(json.dumps(audit(load_frame())))
    elif command == "predict":
        print(json.dumps(predict(json.loads(sys.stdin.read()))))
    elif command == "explore":
        print(json.dumps(explore(json.loads(sys.stdin.read())), default=str))
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
