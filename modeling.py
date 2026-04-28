"""
Train logistic regression, random forest, and XGBoost on the
fatigue features. Train: 2019-2023, test: 2024-2025.
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
)
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


BASE_DIR     = os.path.dirname(__file__)
FEATURE_PATH = os.path.join(BASE_DIR, "data", "features.parquet")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "plots")

# TODO: import from feature_engineering instead of duplicating
FEATURE_COLS = [
    "roll_velocity_mean", "roll_velocity_std", "roll_velocity_slope",
    "velocity_decay", "velocity_drop_max",
    "roll_fb_velocity_mean", "roll_fb_velocity_std",
    "roll_fb_velocity_slope", "fb_velocity_decay",
    "roll_os_velocity_mean", "roll_os_velocity_std",
    "roll_os_velocity_slope", "os_velocity_decay",
    "roll_spin_mean", "roll_spin_std", "roll_spin_slope", "spin_decay",
    "roll_relx_std", "roll_relz_std", "roll_ext_mean", "roll_ext_std",
    "roll_pfxx_mean", "roll_pfxz_mean", "roll_pfxx_std", "roll_pfxz_std",
    "roll_move_mean",
    "roll_platex_std", "roll_platez_std", "roll_in_zone_rate",
    "roll_hc_rate", "hc_rate_rise",
    "roll_xwoba_mean", "xwoba_rise",
    "roll_barrel_rate", "roll_ld_rate",
    "pitch_count", "roll_strike_rate", "eff_speed_decay",
]
TARGET = "fatigue_label"

TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEARS  = [2024, 2025]

RANDOM_STATE = 42


def load_and_split(path: str = FEATURE_PATH):
    log.info(f"Loading features from {path}")
    df = pd.read_parquet(path)

    df["year"] = pd.to_datetime(df["game_date"]).dt.year
    train = df[df["year"].isin(TRAIN_YEARS)]
    test  = df[df["year"].isin(TEST_YEARS)]
    log.info(f"Train: {len(train):,} | Test: {len(test):,}")

    X_train = train[FEATURE_COLS].astype(float)
    y_train = train[TARGET].astype(int)
    X_test  = test[FEATURE_COLS].astype(float)
    y_test  = test[TARGET].astype(int)

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    log.info(f"Positive rate -- train: {y_train.mean():.3%} | test: {y_test.mean():.3%}")
    return X_train, X_test, y_train, y_test


def build_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="saga",
            random_state=RANDOM_STATE,
        )),
    ])


def build_random_forest():
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def build_xgboost(y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def evaluate(name, model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)
    preds = (proba >= threshold).astype(int)
    report = classification_report(y_test, preds, output_dict=True)

    log.info(
        f"{name}: ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} "
        f"P={report['1']['precision']:.3f} R={report['1']['recall']:.3f} "
        f"F1={report['1']['f1-score']:.3f} thr={threshold:.2f}"
    )

    return {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
        "threshold": threshold,
    }


def best_f1_threshold(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    return float(thresholds[np.argmax(f1s[:-1])])


def plot_roc_pr(results, X_test, y_test):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pitcher Fatigue Detection - Model Comparison",
                 fontsize=14, fontweight="bold")

    colors = {"Logistic Regression": "#2196F3",
              "Random Forest":       "#4CAF50",
              "XGBoost":             "#FF5722"}

    for name, model in results.items():
        proba = model.predict_proba(X_test)[:, 1]
        color = colors.get(name, "grey")

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=color, lw=2)

        prec, rec, _ = precision_recall_curve(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})", color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend(loc="lower right")

    baseline_rate = y_test.mean()
    axes[1].axhline(baseline_rate, linestyle="--", color="k", lw=1,
                    label=f"Baseline rate ({baseline_rate:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "roc_pr_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved roc_pr_curves.png")


def plot_feature_importance(model, model_name, feature_names, top_n=20):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
    else:
        clf = model

    if not hasattr(clf, "feature_importances_"):
        return  # logistic regression handled by plot_lr_coefficients

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    feat_names = np.array(feature_names)[idx]
    feat_imp   = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=feat_imp, y=feat_names, palette="viridis", ax=ax)
    ax.set_title(f"{model_name} - Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    fname = model_name.lower().replace(" ", "_") + "_feature_importance.png"
    save_path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved {fname}")


def plot_lr_coefficients(lr_pipeline, feature_names, top_n=20):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    clf = lr_pipeline.named_steps["clf"]
    coef = clf.coef_[0]

    idx = np.argsort(np.abs(coef))[::-1][:top_n]
    names = np.array(feature_names)[idx]
    vals  = coef[idx]
    colors = ["#E53935" if v > 0 else "#1E88E5" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.set_title(f"Logistic Regression - Top {top_n} Coefficients (post-scaling)",
                 fontweight="bold")
    ax.set_xlabel("Coefficient (red = higher fatigue risk, blue = lower)")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "logistic_regression_coefficients.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved logistic_regression_coefficients.png")


def plot_confusion_matrix(model, name, X_test, y_test, threshold=0.5):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Not Fatigued", "Fatigued"],
                yticklabels=["Not Fatigued", "Fatigued"], ax=ax)
    ax.set_title(f"{name} - Normalized Confusion Matrix", fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    fname = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    save_path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved {fname}")


def train(feature_path: str = FEATURE_PATH):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split(feature_path)

    # hold out 2023 as validation for threshold tuning
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )

    log.info("Training Logistic Regression")
    lr = build_logistic_regression()
    lr.fit(X_tr, y_tr)
    lr_thresh = best_f1_threshold(lr, X_val, y_val)
    lr_metrics = evaluate("Logistic Regression", lr, X_test, y_test, threshold=lr_thresh)
    plot_lr_coefficients(lr, FEATURE_COLS)
    plot_confusion_matrix(lr, "Logistic Regression", X_test, y_test, threshold=lr_thresh)
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))

    log.info("Training Random Forest")
    rf = build_random_forest()
    rf.fit(X_tr, y_tr)
    rf_thresh = best_f1_threshold(rf, X_val, y_val)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test, threshold=rf_thresh)
    plot_feature_importance(rf, "Random Forest", FEATURE_COLS)
    plot_confusion_matrix(rf, "Random Forest", X_test, y_test, threshold=rf_thresh)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    log.info("Training XGBoost")
    xgb_model = build_xgboost(y_tr)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    xgb_thresh = best_f1_threshold(xgb_model, X_val, y_val)
    xgb_metrics = evaluate("XGBoost", xgb_model, X_test, y_test, threshold=xgb_thresh)
    plot_feature_importance(xgb_model, "XGBoost", FEATURE_COLS)
    plot_confusion_matrix(xgb_model, "XGBoost", X_test, y_test, threshold=xgb_thresh)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))

    summary = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
    summary_path = os.path.join(BASE_DIR, "results", "model_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved {summary_path}")

    plot_roc_pr({"Logistic Regression": lr,
                 "Random Forest": rf,
                 "XGBoost": xgb_model}, X_test, y_test)

    return summary


def predict_fatigue(model_path, X, threshold=0.5):
    model = joblib.load(model_path)
    X_clean = X[FEATURE_COLS].astype(float).fillna(X[FEATURE_COLS].median())
    proba = model.predict_proba(X_clean)[:, 1]
    return (proba >= threshold).astype(int), proba


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=FEATURE_PATH)
    args = parser.parse_args()

    summary = train(feature_path=args.features)
    print(summary.to_string(index=False))
