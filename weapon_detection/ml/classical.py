"""Step 4: classical models - baseline, Logistic Regression and Random Forest.

Run with:  uv run python -m weapon_detection.ml.classical

Protocol (graded):
  1. Fit every pipeline on the TRAIN split only.
  2. Pick hyperparameters by VALIDATION macro-F1 (at the default 0.5 threshold).
  3. Pick the decision threshold on the VALIDATION scores (max validation macro-F1).
  4. Save the choices to chosen_settings.json, THEN evaluate on TEST exactly once.
"""

import json
import time

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weapon_detection.ml.common import (
    AUDIT_PATH,
    FIGURE_DIR,
    FIREARM_CLASS_NAMES,
    MANIFEST_PATH,
    OUTPUT_DIR,
    REPO_ROOT,
    SEED,
)
from weapon_detection.ml.features import load_features

RESULTS_DIR = OUTPUT_DIR / "results"
MODEL_DIR = REPO_ROOT / "models" / "classical"
SETTINGS_PATH = OUTPUT_DIR / "chosen_settings.json"

# Hyperparameter grids searched on the validation split.
GRIDS = {
    "baseline": [{}],
    "logreg": [{"C": c} for c in [0.001, 0.01, 0.1, 1.0]],
    "random_forest": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in [10, 20, None]
        for leaf in [1, 3]
    ],
}
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)  # candidate decision thresholds
FEATURE_SETS = ["handcrafted", "handcrafted_hog"]
MAIN_FEATURE_SET = "handcrafted_hog"


def make_pipeline(model_name: str, params: dict) -> Pipeline:
    """Build an unfitted scikit-learn pipeline for one model."""
    if model_name == "baseline":
        return Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    if model_name == "logreg":
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED, **params
        )
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
            **params,
        )
        return Pipeline([("model", model)])  # trees do not need scaling
    raise ValueError(f"Unknown model {model_name}")


def firearm_scores(pipeline: Pipeline, x: np.ndarray) -> np.ndarray:
    """Probability of the firearm class (label 1)."""
    column = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(x)[:, column]


def best_threshold(y_val: np.ndarray, val_scores: np.ndarray) -> tuple[float, float]:
    """Threshold that maximises macro-F1 on the VALIDATION data only."""
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f1 = f1_score(y_val, (val_scores >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:  # strict ">" keeps the lowest threshold on ties
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def fit_and_choose(
    model_name: str, x: np.ndarray, y: np.ndarray, split: np.ndarray
) -> tuple[Pipeline, dict]:
    """Tune on train/val only. Test rows are never read (they could even be NaN)."""
    x_train, y_train = x[split == "train"], y[split == "train"]
    x_val, y_val = x[split == "val"], y[split == "val"]

    best_pipeline, best = None, None
    tuning_log = []
    for params in GRIDS[model_name]:
        pipeline = make_pipeline(model_name, params)
        start = time.perf_counter()
        pipeline.fit(x_train, y_train)  # TRAIN split only
        seconds = time.perf_counter() - start
        val_f1 = f1_score(y_val, pipeline.predict(x_val), average="macro", zero_division=0)
        tuning_log.append({"params": params, "val_macro_f1_at_0.5": val_f1, "fit_s": seconds})
        print(f"  {model_name} {params}: val macro-F1 {val_f1:.4f} ({seconds:.1f}s)")
        if best is None or val_f1 > best["val_macro_f1_at_0.5"]:
            best_pipeline = pipeline
            best = {"params": params, "val_macro_f1_at_0.5": val_f1, "train_seconds": seconds}

    if model_name == "baseline":
        best["threshold"] = None  # the baseline just outputs the majority class
        best["val_macro_f1_at_threshold"] = best["val_macro_f1_at_0.5"]
    else:
        threshold, f1 = best_threshold(y_val, firearm_scores(best_pipeline, x_val))
        best["threshold"] = threshold
        best["val_macro_f1_at_threshold"] = f1
    best["tuning_log"] = tuning_log
    return best_pipeline, best


def predict_labels(
    pipeline: Pipeline, x: np.ndarray, scores: np.ndarray, threshold: float | None
) -> np.ndarray:
    """0/1 predictions: scores >= threshold; the baseline (no threshold) uses predict()."""
    if threshold is None:
        return pipeline.predict(x)
    return (scores >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """All metrics, firearm (1) = positive class, plus the no-firearm (0) class separately."""
    both_classes = len(np.unique(y_true)) == 2
    return {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, scores),
        "neg_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "neg_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "neg_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name: str, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["no firearm", "firearm"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_xlabel("predicted label")
    axis.set_ylabel("true label")
    axis.set_title(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / name, dpi=150)
    plt.close(figure)


def max_firearm_area(classes: str, areas: str) -> float:
    """Largest firearm box area fraction in an image (0 if it has none)."""
    if not isinstance(classes, str) or not classes:
        return 0.0
    values = [
        float(a)
        for c, a in zip(classes.split(";"), str(areas).split(";"))
        if c.lower() in FIREARM_CLASS_NAMES
    ]
    return max(values, default=0.0)


def error_analysis(
    manifest: pd.DataFrame, brightness: np.ndarray, run_key: str, out_name: str = "error_analysis"
) -> None:
    """5 most confident false positives and false negatives on TEST (no explanations).

    Writes outputs/ml/<out_name>.csv and figures/<out_name>_grid.png.
    """
    predictions = pd.read_csv(RESULTS_DIR / f"{run_key}_predictions.csv")
    test = predictions[predictions["split"] == "test"].copy()
    info = manifest.set_index("image_path")
    bright = pd.Series(brightness, index=manifest["image_path"])

    false_pos = test[(test["y_true"] == 0) & (test["y_pred"] == 1)]
    false_neg = test[(test["y_true"] == 1) & (test["y_pred"] == 0)]
    false_pos = false_pos.sort_values("score", ascending=False).head(5)  # most sure "firearm"
    false_neg = false_neg.sort_values("score", ascending=True).head(5)  # most sure "no firearm"
    chosen = pd.concat(
        [false_pos.assign(error="false_positive"), false_neg.assign(error="false_negative")]
    )

    table = pd.DataFrame(
        {
            "image": chosen["image"],
            "error": chosen["error"],
            "true_label": chosen["y_true"],
            "score": chosen["score"].round(4),
            "classes_in_image": [info.loc[p, "classes"] for p in chosen["image"]],
            "firearm_box_area_fraction": [
                round(max_firearm_area(info.loc[p, "classes"], info.loc[p, "box_areas"]), 4)
                for p in chosen["image"]
            ],
            "brightness": [round(float(bright[p]), 1) for p in chosen["image"]],
            "my_explanation": "",
        }
    )
    table.to_csv(OUTPUT_DIR / f"{out_name}.csv", index=False)

    figure, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    for axis in axes.flatten():
        axis.axis("off")
    for row, part in enumerate([false_pos, false_neg]):
        for col, (_, item) in enumerate(part.iterrows()):
            image = cv2.imread(str(REPO_ROOT / item["image"]))
            axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            label = "firearm" if item["y_true"] == 1 else "no firearm"
            axes[row, col].set_title(f"true: {label}\nscore: {item['score']:.3f}", fontsize=9)
    figure.suptitle(
        f"Test errors of {run_key}: top row false positives, bottom row false negatives"
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{out_name}_grid.png", dpi=150)
    plt.close(figure)


def write_leakage_audit(feature_names: list[str]) -> None:
    """Markdown table of every feature plus the leakage checks, filled from audit.json."""
    audit = json.loads(AUDIT_PATH.read_text())
    v2 = audit["split_v2"]["summary"]
    sources = audit["split_v2"]["negative_sources_v2"]
    lines = [
        "# Leakage audit",
        "",
        "Generated by `weapon_detection/ml/classical.py` from `outputs/ml/audit.json`.",
        "",
        "| Feature | Available at prediction time? | Removed? | Note |",
        "| --- | --- | --- | --- |",
    ]
    for name in feature_names:
        lines.append(f"| {name} | yes (pixels only) | no | computed from the 128x128 image |")
    lines += [
        "| hog_0 .. hog_1763 | yes (pixels only) | no | HOG of the 128x128 grey image |",
        "| bounding boxes / class names | no (they are the labels) | yes, never used | |",
        "| file name / filename family | no (not meaningful at deployment) | yes, never used | "
        "used only for grouping |",
        "| original split / new split / group id | no | yes, never used | used only for "
        "splitting |",
        "| width / height (original) | yes | yes, not used | images resized to 128x128 |",
        "",
        "| Check | Result |",
        "| --- | --- |",
        "| Split used | v2 (revision made after preliminary v1 test results; see audit.json) |",
        f"| Groups in more than one split | {v2['groups_in_more_than_one_split']} |",
        f"| Exact duplicate pairs (SHA-256) | {audit['exact_duplicate_pairs']} |",
        f"| Near-duplicate pairs (dHash <= {audit['near_duplicate_max_hamming']}) | "
        f"{audit['near_duplicate_pairs']} total, "
        f"{v2['near_duplicate_pairs_across_splits']} across splits |",
        f"| Filename families spanning splits | {v2['families_spanning_splits']} |",
        f"| Distinct negative groups (val / test) | "
        f"{sources['val']['distinct_negative_groups']} / "
        f"{sources['test']['distinct_negative_groups']} |",
        "| Scaler fitted on train only | yes: StandardScaler is inside the Pipeline, "
        "fit() sees train rows only (unit-tested) |",
        "| Hyperparameters chosen on val only | yes (validation macro-F1) |",
        "| Threshold chosen on val only | yes (validation macro-F1), saved to "
        "chosen_settings.json before test |",
        "| Test split used once | yes: one evaluation after settings were saved |",
        "",
    ]
    (OUTPUT_DIR / "leakage_audit.md").write_text("\n".join(lines))


def save_settings(new_settings: dict) -> None:
    """Add or replace entries in chosen_settings.json, keeping entries of other models."""
    settings = json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
    settings.update(new_settings)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def save_metric_rows(rows: list[dict]) -> pd.DataFrame:
    """Add or replace rows in classical_metrics.csv; rows of other models are kept."""
    new = pd.DataFrame(rows)
    path = RESULTS_DIR / "classical_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new["model"])]  # replace rows of the same models
        new = pd.concat([old, new], ignore_index=True)
    new.round(4).to_csv(path, index=False)
    return new


def compare_with_split_v1() -> None:
    """Save test metrics of split v1 next to split v2 (main models and baseline)."""
    v1_path = OUTPUT_DIR / "split_v1" / "results" / "classical_metrics.csv"
    if not v1_path.exists():
        return
    columns = [
        "model",
        "feature_set",
        "macro_f1",
        "neg_precision",
        "neg_recall",
        "neg_f1",
        "balanced_accuracy",
        "accuracy",
        "roc_auc",
        "pr_auc",
    ]
    tables = []
    for version, path in [("v1", v1_path), ("v2", RESULTS_DIR / "classical_metrics.csv")]:
        table = pd.read_csv(path)
        keep = (table["split"] == "test") & (
            (table["model"] == "baseline") | (table["main_model"].astype(bool))
        )
        tables.append(table.loc[keep, columns].assign(split_version=version))
    comparison = pd.concat(tables).sort_values(["model", "split_version"])
    comparison.round(4).to_csv(RESULTS_DIR / "split_v1_vs_v2_test.csv", index=False)
    print("\nTest metrics, split v1 vs split v2:")
    print(comparison.round(4).to_string(index=False))


def main() -> None:
    plt.switch_backend("Agg")
    for folder in [RESULTS_DIR, FIGURE_DIR, MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    handcrafted, hog_matrix, names = load_features(manifest)  # joined by image path
    features = {
        "handcrafted": handcrafted,
        "handcrafted_hog": np.hstack([handcrafted, hog_matrix]),
    }
    y = manifest["firearm_present"].to_numpy()
    split = manifest["split"].to_numpy()
    runs = [("baseline", "handcrafted")] + [
        (model, feature_set)
        for model in ["logreg", "random_forest"]
        for feature_set in FEATURE_SETS
    ]

    # ---- Phase 1: train + validation only -----------------------------------------
    pipelines, settings = {}, {}
    for model_name, feature_set in runs:
        key = f"{model_name}_{feature_set}"
        print(f"Tuning {key} ...")
        pipeline, chosen = fit_and_choose(model_name, features[feature_set], y, split)
        pipelines[key] = pipeline
        settings[key] = chosen
        joblib.dump(pipeline, MODEL_DIR / f"{key}.joblib")
    save_settings(settings)
    print(f"Settings frozen in {SETTINGS_PATH} - now evaluating on test once.\n")

    # ---- Phase 2: evaluate val and test with the frozen settings ---------------------
    rows = []
    for model_name, feature_set in runs:
        key = f"{model_name}_{feature_set}"
        pipeline, threshold = pipelines[key], settings[key]["threshold"]
        prediction_tables = []
        for part in ["val", "test"]:
            mask = split == part
            start = time.perf_counter()
            scores = firearm_scores(pipeline, features[feature_set][mask])
            y_pred = predict_labels(pipeline, features[feature_set][mask], scores, threshold)
            ms_per_image = (time.perf_counter() - start) * 1000 / mask.sum()
            metrics = compute_metrics(y[mask], y_pred, scores)
            rows.append(
                {
                    "model": model_name,
                    "feature_set": "n/a (ignores input)"
                    if model_name == "baseline"
                    else feature_set,
                    "main_model": model_name != "baseline" and feature_set == MAIN_FEATURE_SET,
                    "split": part,
                    "n_images": int(mask.sum()),
                    "threshold": threshold,
                    **metrics,
                    "train_seconds": settings[key]["train_seconds"],
                    "predict_ms_per_image": ms_per_image,
                }
            )
            prediction_tables.append(
                pd.DataFrame(
                    {
                        "image": manifest["image_path"][mask],
                        "split": part,
                        "y_true": y[mask],
                        "score": scores,
                        "y_pred": y_pred,
                    }
                )
            )
            if part == "test":
                save_confusion_matrix(
                    y[mask],
                    y_pred,
                    f"confusion_{key}_test.png",
                    f"{key} - test (threshold {threshold})",
                )
        pd.concat(prediction_tables).to_csv(RESULTS_DIR / f"{key}_predictions.csv", index=False)

    metrics_table = pd.DataFrame(rows)
    save_metric_rows(rows)
    columns = ["model", "feature_set", "split", "macro_f1", "neg_recall", "accuracy", "roc_auc"]
    print(metrics_table[columns].round(4).to_string(index=False))

    # ---- Interpretation aids ----------------------------------------------------------
    for feature_set in FEATURE_SETS:
        coefs = pipelines[f"logreg_{feature_set}"].named_steps["model"].coef_[0][: len(names)]
        table = pd.DataFrame({"feature": names, "coefficient": coefs})
        table = table.reindex(table["coefficient"].abs().sort_values(ascending=False).index)
        table.head(10).round(4).to_csv(
            RESULTS_DIR / f"lr_coefficients_{feature_set}.csv", index=False
        )
    importance_rows = []
    for feature_set in FEATURE_SETS:
        importances = pipelines[f"random_forest_{feature_set}"].named_steps["model"]
        importances = importances.feature_importances_
        for name, value in zip(names, importances[: len(names)]):
            importance_rows.append(
                {"feature_set": feature_set, "feature": name, "importance": value}
            )
        if feature_set == "handcrafted_hog":
            importance_rows.append(
                {
                    "feature_set": feature_set,
                    "feature": "HOG (all 1764, summed)",
                    "importance": importances[len(names) :].sum(),
                }
            )
    pd.DataFrame(importance_rows).round(4).to_csv(RESULTS_DIR / "rf_importance.csv", index=False)

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    test_mask = split == "test"
    for model_name in ["logreg", "random_forest"]:
        for feature_set in FEATURE_SETS:
            key = f"{model_name}_{feature_set}"
            scores = firearm_scores(pipelines[key], features[feature_set][test_mask])
            RocCurveDisplay.from_predictions(y[test_mask], scores, name=key, ax=roc_axis)
            PrecisionRecallDisplay.from_predictions(y[test_mask], scores, name=key, ax=pr_axis)
    roc_axis.set_title("ROC curve (test)")
    pr_axis.set_title("Precision-recall curve (test, firearm = positive)")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "roc_pr_test.png", dpi=150)
    plt.close(figure)

    # ---- Error analysis for the best model by VALIDATION macro-F1 ---------------------
    candidates = [k for k in settings if not k.startswith("baseline")]
    best_key = max(candidates, key=lambda k: settings[k]["val_macro_f1_at_threshold"])
    print(f"\nBest model by validation macro-F1: {best_key}")
    error_analysis(manifest, handcrafted[:, names.index("mean_brightness")], best_key)
    write_leakage_audit(names)
    compare_with_split_v1()
    print(f"Wrote results to {RESULTS_DIR}, figures to {FIGURE_DIR}, models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
