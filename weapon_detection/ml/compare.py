"""Step 6: collect every saved result into final tables, figures and report evidence.

Run with:  uv run python -m weapon_detection.ml.compare

Nothing is retrained and no model is re-run. Metrics are recomputed from the saved
prediction CSVs ONLY to check that they match classical_metrics.csv. The fine-tuned YOLO
row is included automatically once yolo_eval.py has been run on its checkpoint.
"""

import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from weapon_detection.ml.classical import RESULTS_DIR, SETTINGS_PATH, compute_metrics
from weapon_detection.ml.common import (
    AUDIT_PATH,
    EXTRACT_DIR,
    FIGURE_DIR,
    MANIFEST_PATH,
    OUTPUT_DIR,
    REPO_ROOT,
    SEED,
)
from weapon_detection.ml.features import HANDCRAFTED_NAMES, features_for_image

METRICS_PATH = RESULTS_DIR / "classical_metrics.csv"
NUMBERS_PATH = OUTPUT_DIR / "report_numbers.json"
GUI_SMOKE_PATH = OUTPUT_DIR / "gui_smoke_test.json"
FINE_TUNED_CHECKPOINT = REPO_ROOT / "models" / "firearm_yolov8n_best.pt"

# Every model that can appear in the final table, in report order.
# key = prediction-file name; model/feature_set = how the row is stored in the metrics CSV
# (feature_set None means "any"); family = model family named in the report.
RUNS = [
    {
        "key": "baseline_handcrafted",
        "model": "baseline",
        "feature_set": None,
        "name": "Majority baseline",
        "family": "trivial (DummyClassifier)",
        "main": False,
    },
    {
        "key": "logreg_handcrafted",
        "model": "logreg",
        "feature_set": "handcrafted",
        "name": "Logistic Regression (handcrafted)",
        "family": "linear model",
        "main": False,
    },
    {
        "key": "random_forest_handcrafted",
        "model": "random_forest",
        "feature_set": "handcrafted",
        "name": "Random Forest (handcrafted)",
        "family": "tree ensemble",
        "main": False,
    },
    {
        "key": "logreg_handcrafted_hog",
        "model": "logreg",
        "feature_set": "handcrafted_hog",
        "name": "Logistic Regression (handcrafted + HOG)",
        "family": "linear model",
        "main": True,
    },
    {
        "key": "random_forest_handcrafted_hog",
        "model": "random_forest",
        "feature_set": "handcrafted_hog",
        "name": "Random Forest (handcrafted + HOG)",
        "family": "tree ensemble",
        "main": True,
    },
    {
        "key": "yolov8n_oiv7_pretrained",
        "model": "YOLOv8n pretrained (Open Images, not fine-tuned)",
        "feature_set": None,
        "name": "YOLOv8n pretrained (Open Images, not fine-tuned)",
        "family": "CNN object detector",
        "main": False,
    },
    {
        "key": "yolov8n_finetuned",
        "model": "YOLOv8n fine-tuned (firearm)",
        "feature_set": None,
        "name": "YOLOv8n fine-tuned (firearm)",
        "family": "CNN object detector",
        "main": True,
    },
]
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "neg_precision",
    "neg_recall",
    "neg_f1",
    "roc_auc",
    "pr_auc",
]
TABLE_COLUMNS = METRIC_COLUMNS + ["threshold", "predict_ms_per_image"]


# ----------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------
def rows_for_run(metrics: pd.DataFrame, run: dict) -> pd.DataFrame:
    """The metric rows (val and test) belonging to one run; empty if it was never run."""
    mask = metrics["model"] == run["model"]
    if run["feature_set"] is not None:
        mask &= metrics["feature_set"] == run["feature_set"]
    return metrics[mask]


def available_runs(metrics: pd.DataFrame) -> list[dict]:
    """Runs that have metric rows AND a predictions file."""
    found = []
    for run in RUNS:
        has_rows = len(rows_for_run(metrics, run)) > 0
        has_predictions = (RESULTS_DIR / f"{run['key']}_predictions.csv").exists()
        if has_rows and has_predictions:
            found.append(run)
    return found


def check_consistency(
    metrics: pd.DataFrame, runs: list[dict], results_dir: Path = RESULTS_DIR
) -> list[str]:
    """Recompute metrics from each predictions CSV and compare with the saved table.

    The table is stored rounded to 4 decimals, so the recomputed value is rounded the same
    way before comparing; the allowed difference is 1e-9. Returns a list of mismatches.
    """
    mismatches = []
    for run in runs:
        predictions = pd.read_csv(results_dir / f"{run['key']}_predictions.csv")
        for _, row in rows_for_run(metrics, run).iterrows():
            part = predictions[predictions["split"] == row["split"]]
            if len(part) != row["n_images"]:
                mismatches.append(f"{run['key']} {row['split']}: n_images differs")
            recomputed = compute_metrics(
                part["y_true"].to_numpy(), part["y_pred"].to_numpy(), part["score"].to_numpy()
            )
            for name in METRIC_COLUMNS:
                new = round(float(recomputed[name]), 4)
                old = float(row[name])
                if not (abs(new - old) <= 1e-9 or (np.isnan(new) and np.isnan(old))):
                    mismatches.append(f"{run['key']} {row['split']} {name}: {old} vs {new}")
    return mismatches


def final_table(metrics: pd.DataFrame, runs: list[dict], split: str) -> pd.DataFrame:
    """One row per model for one split, in report order."""
    rows = []
    for run in runs:
        row = rows_for_run(metrics, run)
        row = row[row["split"] == split].iloc[0]
        rows.append(
            {
                "model": run["name"],
                "family": run["family"],
                "main_model": "yes" if run["main"] else "",
                **{c: row[c] for c in TABLE_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    """A simple GitHub markdown table (no extra dependency)."""

    def cell(value) -> str:
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(table.columns) + " |", "|" + " --- |" * len(table.columns)]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def detection_table() -> pd.DataFrame:
    """Detection metrics (mAP etc.) from *_detection_metrics.json, kept in their own table."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*_detection_metrics.json")):
        data = json.loads(path.read_text())
        rows.append({"checkpoint_key": path.name.replace("_detection_metrics.json", ""), **data})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------
def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / name, dpi=150)
    plt.close(figure)


def plot_negative_sources(audit: dict, name: str) -> None:
    v1 = audit["split_v2"]["negative_sources_v1"]
    v2 = audit["split_v2"]["negative_sources_v2"]
    splits = ["train", "val", "test"]
    x = np.arange(len(splits))
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for offset, version_name, data in [(-0.2, "split v1", v1), (0.2, "split v2", v2)]:
        groups = [data[s]["distinct_negative_groups"] for s in splits]
        shares = [data[s]["largest_negative_group_share"] * 100 for s in splits]
        left.bar_label(left.bar(x + offset, groups, width=0.4, label=version_name))
        bars = right.bar(x + offset, shares, width=0.4, label=version_name)
        right.bar_label(bars, fmt="%.1f")
    left.set(
        xticks=x,
        xticklabels=splits,
        xlabel="split",
        ylabel="distinct negative groups",
        title="Number of source groups the negatives come from",
    )
    right.set(
        xticks=x,
        xticklabels=splits,
        xlabel="split",
        ylabel="largest group's share of negatives (%)",
        title="Share of negatives from the single largest group",
    )
    left.legend()
    right.legend()
    save_figure(figure, name)


def plot_confusion_matrices(runs: list[dict], name: str) -> None:
    main_runs = [r for r in runs if r["main"]]
    figure, axes = plt.subplots(1, len(main_runs), figsize=(4.5 * len(main_runs), 4.2))
    axes = np.atleast_1d(axes)
    for axis, run in zip(axes, main_runs):
        predictions = pd.read_csv(RESULTS_DIR / f"{run['key']}_predictions.csv")
        test = predictions[predictions["split"] == "test"]
        ConfusionMatrixDisplay.from_predictions(
            test["y_true"],
            test["y_pred"],
            labels=[0, 1],
            display_labels=["no firearm", "firearm"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(run["name"], fontsize=9)
        axis.set_xlabel("predicted label")
        axis.set_ylabel("true label")
    figure.suptitle("Test confusion matrices of the main models")
    save_figure(figure, name)


def plot_curves(runs: list[dict], roc_name: str, pr_name: str) -> None:
    roc_figure, roc_axis = plt.subplots(figsize=(6.5, 5))
    pr_figure, pr_axis = plt.subplots(figsize=(6.5, 5))
    for run in runs:
        predictions = pd.read_csv(RESULTS_DIR / f"{run['key']}_predictions.csv")
        test = predictions[predictions["split"] == "test"]
        RocCurveDisplay.from_predictions(
            test["y_true"], test["score"], name=run["name"], ax=roc_axis
        )
        PrecisionRecallDisplay.from_predictions(
            test["y_true"], test["score"], name=run["name"], ax=pr_axis
        )
    roc_axis.plot([0, 1], [0, 1], "k:", linewidth=1)
    roc_axis.set(
        title="ROC curves on the test split",
        xlabel="false positive rate",
        ylabel="true positive rate",
    )
    pr_axis.set(
        title="Precision-recall curves on the test split (firearm = positive)",
        xlabel="recall",
        ylabel="precision",
    )
    for axis in [roc_axis, pr_axis]:
        axis.legend(fontsize=7)
    save_figure(roc_figure, roc_name)
    save_figure(pr_figure, pr_name)


def plot_macro_f1_bars(test_table: pd.DataFrame, name: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(test_table))
    axis.barh(y - 0.2, test_table["macro_f1"], height=0.4, label="macro-F1")
    axis.barh(y + 0.2, test_table["neg_recall"], height=0.4, label="no-firearm recall")
    for i, (f1, rec) in enumerate(zip(test_table["macro_f1"], test_table["neg_recall"])):
        axis.text(f1 + 0.01, i - 0.2, f"{f1:.3f}", va="center", fontsize=8)
        axis.text(rec + 0.01, i + 0.2, f"{rec:.3f}", va="center", fontsize=8)
    axis.set(
        yticks=y,
        yticklabels=test_table["model"],
        xlim=(0, 1.1),
        xlabel="score",
        title="Test macro-F1 and no-firearm (negative-class) recall per model",
    )
    axis.invert_yaxis()
    axis.legend(loc="lower right")
    save_figure(figure, name)


def plot_latency(test_table: pd.DataFrame, runs: list[dict], feature_ms: float, name: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(test_table))
    is_classical = [not r["family"].startswith("CNN") for r in runs]
    model_ms = test_table["predict_ms_per_image"].to_numpy()
    extraction = np.where(is_classical, feature_ms, 0.0)
    # Two bars per model on a log axis: feature extraction and model prediction.
    axis.barh(y - 0.2, extraction, height=0.4, label="feature extraction (classical only)")
    axis.barh(y + 0.2, model_ms, height=0.4, label="model prediction (YOLO: whole pipeline)")
    for i, (ext, mod) in enumerate(zip(extraction, model_ms)):
        if ext > 0:
            axis.text(ext * 1.2, i - 0.2, f"{ext:.2f} ms", va="center", fontsize=8)
        axis.text(mod * 1.2, i + 0.2, f"{mod:.4f} ms", va="center", fontsize=8)
    axis.set_xscale("log")
    axis.set(
        yticks=y,
        yticklabels=test_table["model"],
        xlabel="milliseconds per image (log)",
        title="CPU latency per test image",
    )
    axis.invert_yaxis()
    axis.set_xlim(1e-4, 5e3)
    axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save_figure(figure, name)


def time_feature_extraction(manifest: pd.DataFrame, n_images: int = 50) -> float:
    """Mean ms to read one test image and compute its features (50 seeded test images)."""
    test = manifest[manifest["split"] == "test"].sample(n=n_images, random_state=SEED)
    start = time.perf_counter()
    for path in test["image_path"]:
        features_for_image(REPO_ROOT / path)
    return (time.perf_counter() - start) * 1000 / n_images


# ----------------------------------------------------------------------------------------
# Requirement checklist
# ----------------------------------------------------------------------------------------
def notebook_has_outputs(path: Path) -> bool:
    if not path.exists():
        return False
    cells = json.loads(path.read_text())["cells"]
    return any(c.get("outputs") for c in cells if c["cell_type"] == "code")


def requirement_rows(facts: dict) -> list[dict]:
    """Strict met / partly / not yet status for each course requirement."""

    def row(requirement, ok, partly, evidence, note=""):
        status = "met" if ok else ("partly" if partly else "not yet")
        return {"requirement": requirement, "status": status, "evidence": evidence, "note": note}

    trained = facts["trained_model_families"]  # {model name: family}, fitted on this data
    families = sorted(set(trained.values()))
    notebooks = facts["notebooks_executed"]
    return [
        row(
            ">= 500 images",
            facts["n_images"] >= 500,
            False,
            "outputs/ml/audit.json",
            f"{facts['n_images']} images",
        ),
        row(
            ">= 6 meaningful features",
            facts["n_handcrafted"] >= 6,
            False,
            "weapon_detection/ml/features.py",
            f"{facts['n_handcrafted']} handcrafted + {facts['n_hog']} HOG values",
        ),
        row(
            "trivial baseline",
            facts["has_baseline"],
            False,
            "outputs/ml/results/final_test_table.md",
            "DummyClassifier(most_frequent)",
        ),
        row(
            ">= 3 models from >= 2 families (>= 1 classical)",
            len(trained) >= 3 and len(families) >= 2,
            len(trained) >= 2,
            "outputs/ml/results/final_test_table.md",
            f"trained on this data: {', '.join(f'{m} ({f})' for m, f in trained.items())}"
            + (
                ""
                if len(trained) >= 3
                else "; YOLOv8n pretrained is evaluated but not "
                "trained on this data; fine-tuned YOLO pending"
            ),
        ),
        row(
            "train/test split decided before modelling",
            False,
            True,
            "outputs/ml/audit.json (split_v2.note); weapon_detection/ml/split_v2.py",
            "split fixed before modelling, but revised once (v2) after preliminary v1 "
            "test results; both versions documented",
        ),
        row(
            "all preprocessing fitted on the training split only",
            True,
            False,
            "weapon_detection/ml/classical.py; tests/test_ml_classical.py",
            "StandardScaler inside Pipeline; per-image features need no fitting",
        ),
        row(
            "precision / recall / F1 + confusion matrix",
            facts["has_confusion"],
            False,
            "outputs/ml/results/final_test_table.md; outputs/ml/figures/fig_07_*",
        ),
        row(
            "leakage audit",
            facts["has_leakage_audit"],
            False,
            "outputs/ml/leakage_audit.md; outputs/ml/audit.json",
        ),
        row(
            "EDA with a one-line takeaway per figure",
            False,
            facts["has_eda"],
            "outputs/ml/eda_notes.md; outputs/ml/figure_index.md",
            "figures and factual notes exist; takeaways are for the student to write",
        ),
        row(
            "error analysis (5-10 wrong predictions)",
            facts["error_explanations_written"],
            facts["has_error_tables"],
            "outputs/ml/error_analysis*.csv; fig_12+",
            "wrong predictions listed; 'my_explanation' column still to be written"
            if not facts["error_explanations_written"]
            else "",
        ),
        row(
            "GUI that takes audience input and predicts live",
            facts["gui_smoke_ok"],
            False,
            "predict_app.py; outputs/ml/gui_smoke_test.json",
        ),
        row("fixed random seeds", True, False, "weapon_detection/ml/common.py (SEED = 42)"),
        row(
            "documented notebooks",
            all(notebooks.values()),
            any(notebooks.values()),
            "notebooks/",
            ", ".join(
                f"{n}: {'outputs stored' if ok else 'no stored outputs'}"
                for n, ok in notebooks.items()
            ),
        ),
        row(
            "runs on CPU or free Colab",
            True,
            False,
            "README.md (ML experiment); notebooks/03_yolo_colab.ipynb",
        ),
        row(
            "AI tools listed honestly in the report",
            False,
            False,
            "outputs/report/ai_use_disclosure_facts.md",
            "student to confirm and write",
        ),
    ]


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main() -> None:
    plt.switch_backend("Agg")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)
    settings = json.loads(SETTINGS_PATH.read_text())
    audit = json.loads(AUDIT_PATH.read_text())
    manifest = pd.read_csv(MANIFEST_PATH)

    runs = available_runs(metrics)
    keys = [r["key"] for r in runs]
    if "yolov8n_finetuned" not in keys:
        print("fine-tuned YOLO not evaluated yet")

    # 1. Consistency check against the saved predictions.
    mismatches = check_consistency(metrics, runs)
    for line in mismatches:
        print("MISMATCH:", line)
    assert not mismatches, "Saved metrics do not match the saved predictions"
    print(f"Consistency check: {len(runs)} models x 2 splits recomputed, 0 mismatches")

    # 2. Final tables.
    tables = {}
    for split in ["test", "val"]:
        table = final_table(metrics, runs, split)
        tables[split] = table
        table.round(4).to_csv(RESULTS_DIR / f"final_{split}_table.csv", index=False)
        title = f"# Image-level results on the {split.upper()} split\n\n"
        note = (
            "Firearm = positive class. neg_* = no-firearm class. Threshold chosen on "
            "validation. ms/image: classical = model only on precomputed features; "
            "YOLO = read + inference.\n\n"
        )
        (RESULTS_DIR / f"final_{split}_table.md").write_text(title + note + markdown_table(table))
    print(
        tables["test"][["model", "macro_f1", "neg_recall", "accuracy", "roc_auc"]]
        .round(4)
        .to_string(index=False)
    )

    detection = detection_table()
    detection_md = "# Detection metrics (separate from image-level metrics)\n\n"
    if len(detection):
        detection.round(4).to_csv(RESULTS_DIR / "detection_metrics.csv", index=False)
        detection_md += markdown_table(detection)
    else:
        detection_md += "No detection metrics yet (fine-tuned YOLO not evaluated).\n"
    (RESULTS_DIR / "detection_metrics.md").write_text(detection_md)

    # 3. Figures, numbered in report order. Old fig_* files are removed first.
    for old in FIGURE_DIR.glob("fig_*.png"):
        old.unlink()
    feature_ms = time_feature_extraction(manifest)
    best_classical = max(
        [k for k in settings if k.startswith(("logreg", "random_forest"))],
        key=lambda k: settings[k]["val_macro_f1_at_threshold"],
    )
    figures = []  # (filename, caption)

    def copy_figure(source: str, target: str, caption: str) -> None:
        if (FIGURE_DIR / source).exists():
            shutil.copy(FIGURE_DIR / source, FIGURE_DIR / target)
            figures.append((target, caption))

    copy_figure(
        "class_balance_per_split.png",
        "fig_01_class_balance.png",
        "Number of firearm and no-firearm images in train, val and test (split v2).",
    )
    copy_figure(
        "object_class_counts.png",
        "fig_02_object_classes.png",
        "Annotated objects per original class, stacked by split.",
    )
    copy_figure(
        "firearm_box_area_hist.png",
        "fig_03_firearm_box_area.png",
        "Histogram of firearm box area as a fraction of the image (train).",
    )
    copy_figure(
        "image_size_distribution.png",
        "fig_04_image_sizes.png",
        "The ten most common original image sizes (train).",
    )
    copy_figure(
        "feature_distributions.png",
        "fig_05_feature_distributions.png",
        "Distributions of four handcrafted features by class (train, density).",
    )
    plot_negative_sources(audit, "fig_06_negative_sources_v1_v2.png")
    figures.append(
        (
            "fig_06_negative_sources_v1_v2.png",
            "Distinct negative source groups and largest-group share per split, "
            "split v1 vs split v2.",
        )
    )
    plot_confusion_matrices(runs, "fig_07_confusion_main_models_test.png")
    figures.append(
        (
            "fig_07_confusion_main_models_test.png",
            "Test confusion matrices (counts) of the main models.",
        )
    )
    plot_curves(runs, "fig_08_roc_test.png", "fig_09_pr_test.png")
    figures.append(("fig_08_roc_test.png", "ROC curves of all models on the test split."))
    figures.append(
        ("fig_09_pr_test.png", "Precision-recall curves of all models on the test split.")
    )
    plot_macro_f1_bars(tables["test"], "fig_10_macro_f1_neg_recall_test.png")
    figures.append(
        (
            "fig_10_macro_f1_neg_recall_test.png",
            "Test macro-F1 and no-firearm recall for every model.",
        )
    )
    plot_latency(tables["test"], runs, feature_ms, "fig_11_latency.png")
    figures.append(
        (
            "fig_11_latency.png",
            f"CPU milliseconds per test image; classical bars include "
            f"{feature_ms:.1f} ms feature extraction (mean of 50 test images).",
        )
    )
    copy_figure(
        "error_analysis_grid.png",
        "fig_12_errors_best_classical.png",
        f"Five most confident false positives (top) and false negatives (bottom) "
        f"on test for {best_classical}.",
    )
    copy_figure(
        "error_analysis_yolov8n_oiv7_pretrained_grid.png",
        "fig_13_errors_yolov8n_pretrained.png",
        "Five most confident false positives and false negatives on test for "
        "YOLOv8n pretrained (Open Images).",
    )
    copy_figure(
        "error_analysis_yolov8n_finetuned_grid.png",
        "fig_14_errors_yolov8n_finetuned.png",
        "Five most confident false positives and false negatives on test for YOLOv8n fine-tuned.",
    )

    index = [
        "# Figure index",
        "",
        "Captions describe what is plotted. Takeaways are written by the student.",
        "",
    ]
    for number, (filename, caption) in enumerate(figures, start=1):
        index += [
            f"## Figure {number}: `{filename}`",
            "",
            f"Caption: {caption}",
            "",
            "TAKEAWAY: <student to write>",
            "",
        ]
    (OUTPUT_DIR / "figure_index.md").write_text("\n".join(index))

    # 4. Requirement checklist.
    error_files = sorted(OUTPUT_DIR.glob("error_analysis*.csv"))
    explanations = [pd.read_csv(p)["my_explanation"] for p in error_files]
    gui = json.loads(GUI_SMOKE_PATH.read_text()) if GUI_SMOKE_PATH.exists() else {}
    trained = {
        r["name"].split(" (")[0]: r["family"]
        for r in runs
        if r["family"] in ("linear model", "tree ensemble")
    }
    if "yolov8n_finetuned" in keys:
        trained["YOLOv8n fine-tuned"] = "CNN object detector"
    facts = {
        "n_images": audit["images_used"],
        "n_handcrafted": len(HANDCRAFTED_NAMES),
        "n_hog": 1764,
        "has_baseline": "baseline_handcrafted" in keys,
        "trained_model_families": trained,
        "has_confusion": (FIGURE_DIR / "fig_07_confusion_main_models_test.png").exists(),
        "has_leakage_audit": (OUTPUT_DIR / "leakage_audit.md").exists(),
        "has_eda": (OUTPUT_DIR / "eda_notes.md").exists(),
        "has_error_tables": bool(error_files),
        "error_explanations_written": bool(explanations)
        and all(e.notna().all() and (e.astype(str).str.strip() != "").all() for e in explanations),
        "gui_smoke_ok": gui.get("health_status") == 200 and gui.get("apptest_exceptions") == 0,
        "notebooks_executed": {
            name: notebook_has_outputs(REPO_ROOT / "notebooks" / name)
            for name in [
                "01_data_and_eda.ipynb",
                "02_classical_models.ipynb",
                "03_yolo_colab.ipynb",
                "04_comparison.ipynb",
            ]
        },
    }
    requirements = pd.DataFrame(requirement_rows(facts))
    (OUTPUT_DIR / "requirements_check.md").write_text(
        "# Course requirement checklist\n\nGenerated by `weapon_detection/ml/compare.py`.\n\n"
        + markdown_table(requirements)
    )
    print(requirements[["requirement", "status"]].to_string(index=False))

    # 5. Every number the report cites, with its source file.
    numbers = build_report_numbers(audit, settings, tables, detection, feature_ms, runs)
    NUMBERS_PATH.write_text(json.dumps(numbers, indent=2, default=str))

    # 6. Report evidence pack (outlines, facts, presenter notes; no conclusions).
    from weapon_detection.ml.report_pack import write_report_pack

    write_report_pack(numbers, figures, runs, best_classical)
    print(f"Wrote tables, {len(figures)} figures, report_numbers.json and outputs/report/")


def build_report_numbers(audit, settings, tables, detection, feature_ms, runs) -> dict:
    """Collect cited numbers; each block names the file it came from."""
    data_yaml = yaml.safe_load((EXTRACT_DIR / "data.yaml").read_text())
    readme = EXTRACT_DIR / "README.dataset.txt"
    license_lines = [ln for ln in readme.read_text().splitlines() if "License" in ln]
    yolo_summary_path = REPO_ROOT / "outputs" / "yolo_firearm" / "export_summary.json"
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "opencv-python",
        "matplotlib",
        "joblib",
        "torch",
        "ultralytics",
        "streamlit",
    ]
    return {
        "seed": {"source": "weapon_detection/ml/common.py", "value": SEED},
        "license": {
            "source": "datasets/raw/weapon_detection_extracted/data.yaml + README.dataset.txt",
            "data_yaml_license": data_yaml.get("roboflow", {}).get("license"),
            "readme_license_lines": license_lines,
            "roboflow_url": data_yaml.get("roboflow", {}).get("url"),
        },
        "dataset": {
            "source": "outputs/ml/audit.json",
            "class_names": audit["class_names"],
            "firearm_classes": audit["firearm_classes"],
            "images": audit["images_used"],
            "positives": audit["positives"],
            "negatives": audit["negatives"],
            "hard_negatives_with_other_objects": audit["hard_negatives_with_other_objects"],
            "no_box_images": audit["no_box_images"],
            "per_original_split": audit["per_original_split"],
        },
        "leakage_audit": {
            "source": "outputs/ml/audit.json",
            "same_source_pairs": audit["same_source_pairs"],
            "exact_duplicate_pairs": audit["exact_duplicate_pairs"],
            "near_duplicate_pairs": audit["near_duplicate_pairs"],
            "near_duplicate_pairs_crossing_original_split": audit[
                "near_duplicate_pairs_crossing_original_split"
            ],
            "near_duplicate_label_conflicts": audit["near_duplicate_label_conflicts"],
            "families": audit["families"],
            "before_family_fix": audit["before_family_fix"],
            "after_family_fix_split_v1": audit["after_family_fix"],
            "split_v2": audit["split_v2"],
        },
        "chosen_settings": {
            "source": "outputs/ml/chosen_settings.json",
            "values": {
                k: {
                    x: v.get(x)
                    for x in [
                        "params",
                        "threshold",
                        "val_macro_f1_at_threshold",
                        "train_seconds",
                        "imgsz",
                    ]
                }
                for k, v in settings.items()
            },
        },
        "test_metrics": {
            "source": "outputs/ml/results/final_test_table.csv",
            "rows": tables["test"].round(4).to_dict(orient="records"),
        },
        "val_metrics": {
            "source": "outputs/ml/results/final_val_table.csv",
            "rows": tables["val"].round(4).to_dict(orient="records"),
        },
        "split_v1_vs_v2_test": {
            "source": "outputs/ml/results/split_v1_vs_v2_test.csv",
            "rows": pd.read_csv(RESULTS_DIR / "split_v1_vs_v2_test.csv").to_dict(orient="records"),
        },
        "detection_metrics": {
            "source": "outputs/ml/results/detection_metrics.csv",
            "rows": detection.to_dict(orient="records"),
        },
        "timing": {
            "source": "outputs/ml/results/final_test_table.csv + compare.py timing",
            "feature_extraction_ms_per_image": round(feature_ms, 2),
            "predict_ms_per_image": dict(
                zip(tables["test"]["model"], tables["test"]["predict_ms_per_image"].round(4))
            ),
            "train_seconds": {k: v.get("train_seconds") for k, v in settings.items()},
        },
        "yolo_export": {
            "source": "outputs/yolo_firearm/export_summary.json",
            "summary": json.loads(yolo_summary_path.read_text())
            if yolo_summary_path.exists()
            else None,
        },
        "models_in_table": [r["name"] for r in runs],
        "fine_tuned_yolo_included": any(r["key"] == "yolov8n_finetuned" for r in runs),
        "fine_tuned_checkpoint_exists": FINE_TUNED_CHECKPOINT.exists(),
        "package_versions": {
            "source": "importlib.metadata",
            "values": {p: version(p) for p in packages},
        },
    }


if __name__ == "__main__":
    main()
