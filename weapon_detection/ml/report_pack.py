"""Write the report evidence pack to outputs/report/ (called by compare.py).

Only facts, file pointers and placeholders are written here. Interpretations, explanations,
discussion and conclusions are left for the student ("STUDENT TO WRITE").
"""

from pathlib import Path

import pandas as pd

from weapon_detection.ml.classical import RESULTS_DIR
from weapon_detection.ml.common import FIREARM_CLASS_NAMES, MANIFEST_PATH, REPO_ROOT

REPORT_DIR = REPO_ROOT / "outputs" / "report"
NAMES = {
    "baseline_handcrafted": "Majority baseline",
    "logreg_handcrafted_hog": "Logistic Regression (handcrafted + HOG)",
    "random_forest_handcrafted_hog": "Random Forest (handcrafted + HOG)",
    "yolov8n_oiv7_pretrained": "YOLOv8n pretrained (Open Images)",
    "yolov8n_finetuned": "YOLOv8n fine-tuned",
}


def line_of(relative_path: str, snippet: str) -> str:
    """'path:line' of the first line containing snippet (so pointers are always real)."""
    lines = (REPO_ROOT / relative_path).read_text().splitlines()
    for number, line in enumerate(lines, start=1):
        if snippet in line:
            return f"{relative_path}:{number}"
    raise ValueError(f"'{snippet}' not found in {relative_path}")


def figure_number(figures: list[tuple[str, str]], prefix: str) -> str:
    for number, (filename, _) in enumerate(figures, start=1):
        if filename.startswith(prefix):
            return f"Figure {number} ({filename})"
    return f"(figure {prefix} not generated)"


def demo_examples(best_key: str) -> dict:
    """One true positive, one true negative and one error from TEST for the demo."""
    predictions = pd.read_csv(RESULTS_DIR / f"{best_key}_predictions.csv")
    test = predictions[predictions["split"] == "test"]
    correct = test[test["y_true"] == test["y_pred"]]
    examples = {
        "true positive (highest score)": correct[correct["y_true"] == 1]
        .sort_values("score", ascending=False)
        .iloc[0],
        "true negative (lowest score)": correct[correct["y_true"] == 0]
        .sort_values("score")
        .iloc[0],
    }
    false_pos = test[(test["y_true"] == 0) & (test["y_pred"] == 1)]
    if len(false_pos):
        examples["error: most confident false positive"] = false_pos.sort_values(
            "score", ascending=False
        ).iloc[0]
    return examples


def write_outline(figures, numbers, report_dir: Path) -> None:
    f = lambda prefix: figure_number(figures, prefix)  # noqa: E731 - short local helper
    sections = [
        (
            "1. Task, dataset and why it matters",
            1.0,
            [
                "outputs/ml/report_numbers.json (dataset, license)",
                "docs/DATASET.md",
                "outputs/ml/audit.json (per_original_split)",
            ],
            [f("fig_02")],
            [
                "What is the task (firearm present yes/no) and who would use it?",
                "Describe the dataset source, licence and size in your own words.",
                "Why does this problem matter, and what are the risks of errors?",
            ],
        ),
        (
            "2. Pre-processing",
            1.25,
            [
                "weapon_detection/ml/prepare_data.py",
                "weapon_detection/ml/features.py",
                "outputs/ml/audit.json",
                "outputs/ml/leakage_audit.md",
            ],
            [],
            [
                "How labels were derived (Gun/Pistol -> firearm; knife etc. -> negative).",
                "Duplicate / near-duplicate / video-family grouping and why it was needed.",
                "Feature extraction (128x128 resize, 10 handcrafted features, HOG).",
            ],
        ),
        (
            "3. Exploratory data analysis",
            1.25,
            ["outputs/ml/eda_notes.md", "outputs/ml/figure_index.md"],
            [f("fig_01"), f("fig_02"), f("fig_03"), f("fig_04"), f("fig_05")],
            ["One takeaway per figure (fill the TAKEAWAY lines in figure_index.md)."],
        ),
        (
            "4. Models compared and validation strategy",
            1.5,
            [
                "weapon_detection/ml/split_v2.py",
                "weapon_detection/ml/classical.py",
                "outputs/ml/chosen_settings.json",
                "outputs/ml/split_v1/ (appendix)",
            ],
            [f("fig_06")],
            [
                "Why a group-aware split; why split v1 was revised (state that it was revised "
                "after seeing preliminary v1 test results).",
                "Models and families; hyperparameter grids; threshold chosen on validation.",
                "How YOLO scores are turned into an image-level decision.",
            ],
        ),
        (
            "5. Results table (including baseline)",
            1.5,
            [
                "outputs/ml/results/final_test_table.md",
                "outputs/ml/results/final_val_table.md",
                "outputs/ml/results/detection_metrics.md",
                "outputs/ml/results/split_v1_vs_v2_test.csv",
            ],
            [f("fig_07"), f("fig_08"), f("fig_09"), f("fig_10"), f("fig_11")],
            [
                "Compare each model with the baseline using macro-F1 and no-firearm recall.",
                "Keep detection metrics (mAP) in a separate table.",
            ],
        ),
        (
            "6. Error analysis (5-10 wrong predictions)",
            1.5,
            [
                "outputs/ml/error_analysis.csv",
                "outputs/ml/error_analysis_yolov8n_oiv7_pretrained.csv",
            ],
            [f("fig_12"), f("fig_13"), f("fig_14")],
            [
                "Fill the 'my_explanation' column yourself for each wrong prediction.",
                "Group the errors into patterns you observe.",
            ],
        ),
        (
            "7. Techniques from class",
            1.0,
            ["outputs/report/report_outline.md (technique list below)"],
            [],
            ["Reference at least 2 lecture weeks in depth and at least 3 overall."],
        ),
        (
            "8. Challenges and limitations",
            1.0,
            ["outputs/report/limitations_facts.md"],
            [],
            ["Which limitations matter most for a real deployment, and why?"],
        ),
    ]
    techniques = [
        (
            "group-aware train/validation/test split",
            line_of("weapon_detection/ml/split_v2.py", "def make_split_v2"),
        ),
        (
            "bias-variance trade-off via hyperparameter tuning (max_depth, min_samples_leaf, C)",
            line_of("weapon_detection/ml/classical.py", "GRIDS = {"),
        ),
        (
            "regularisation strength C in Logistic Regression",
            line_of("weapon_detection/ml/classical.py", '"logreg": [{"C": c}'),
        ),
        (
            "ensembles: Random Forest",
            line_of("weapon_detection/ml/classical.py", "model = RandomForestClassifier("),
        ),
        (
            "evaluation metrics (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix)",
            line_of("weapon_detection/ml/classical.py", "def compute_metrics"),
        ),
        (
            "class imbalance (class_weight, macro-F1, threshold tuning)",
            line_of("weapon_detection/ml/classical.py", "def best_threshold"),
        ),
        (
            "feature scaling (StandardScaler fitted on train only)",
            line_of("weapon_detection/ml/classical.py", '("scaler", StandardScaler())'),
        ),
        (
            "feature engineering (handcrafted features, HOG)",
            line_of("weapon_detection/ml/features.py", "def hog_features"),
        ),
        (
            "CNN / object detection with transfer learning (YOLOv8n)",
            line_of("weapon_detection/ml/yolo_train.py", "YOLO(args.model).train"),
        ),
    ]
    lines = [
        "# Report outline (evidence pack)",
        "",
        "Ten-page budget in brackets. Every number must come from "
        "`outputs/ml/report_numbers.json`. Prose is written by the student.",
        "",
    ]
    for title, pages, files, figs, prompts in sections:
        lines += [f"## {title} [{pages} pages]", "", "Evidence files:"]
        lines += [f"- {x}" for x in files]
        if figs:
            lines += ["", "Figures:"] + [f"- {x}" for x in figs]
        lines += [""] + [f"STUDENT TO WRITE: {p}" for p in prompts] + [""]
    lines += ["## Techniques actually used in code", ""]
    for name, where in techniques:
        lines += [
            f"- {name} - code: `{where}`",
            f"  - STUDENT TO FILL: which lecture week covers {name}",
        ]
    (report_dir / "report_outline.md").write_text("\n".join(lines) + "\n")


def write_limitations(numbers, report_dir: Path) -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    sources = numbers["leakage_audit"]["split_v2"]["negative_sources_v2"]
    negatives = manifest[manifest["firearm_present"] == 0]
    negative_classes = negatives["classes"].fillna("").str.split(";").explode()
    negative_classes = negative_classes[negative_classes != ""].value_counts()
    in_family = int((negatives["family"].fillna("") != "").sum())
    train = manifest[manifest["split"] == "train"]
    areas = []
    for classes, box_areas in zip(train["classes"].fillna(""), train["box_areas"].fillna("")):
        for name, area in zip(str(classes).split(";"), str(box_areas).split(";")):
            if name.lower() in FIREARM_CLASS_NAMES:
                areas.append(float(area))
    small = sum(a < 0.05 for a in areas) / len(areas)
    timing = numbers["timing"]
    yolo_threshold = (
        numbers["chosen_settings"]["values"].get("yolov8n_oiv7_pretrained", {}).get("threshold")
    )
    facts = [
        "Positive share per split (v2): "
        + ", ".join(
            f"{s} {sources[s]['positives'] / sources[s]['images']:.1%}"
            for s in ["train", "val", "test"]
        )
        + ".",
        f"Negative images: val {sources['val']['negatives']}, test {sources['test']['negatives']}.",
        f"{numbers['dataset']['negatives']} negative images in total; "
        f"{numbers['dataset']['hard_negatives_with_other_objects']} contain other annotated "
        f"objects and {numbers['dataset']['no_box_images']} have no boxes.",
        "Objects inside negative images: "
        + ", ".join(f"{k} {v}" for k, v in negative_classes.items())
        + ".",
        f"{in_family} negative images belong to filename families (video-frame sequences "
        f"or photo series).",
        f"All images come from one Roboflow project ({numbers['license']['roboflow_url']}).",
        f"{small:.1%} of {len(areas)} train firearm boxes cover less than 5% of the image.",
        "Classical models resize every image to 128x128 regardless of aspect ratio.",
        "The split was revised (v1 -> v2) after preliminary v1 test results were seen; "
        "v1 results are kept in outputs/ml/split_v1/.",
        "Hyperparameters and thresholds were chosen on the validation split only.",
        f"YOLOv8n pretrained: validation-chosen threshold {yolo_threshold} lies below the "
        f"0.05 confidence floor used for prediction.",
        f"Classical feature extraction takes {timing['feature_extraction_ms_per_image']} ms "
        f"per image on this CPU; YOLO latencies are in final_test_table.csv.",
        "No evaluation on real CCTV footage or video streams was performed.",
        "Fine-tuned YOLO: "
        + ("evaluated." if numbers["fine_tuned_yolo_included"] else "not evaluated yet."),
    ]
    text = "# Limitations (facts only)\n\nComputed by `weapon_detection/ml/report_pack.py`.\n\n"
    text += "\n".join(f"- {x}" for x in facts) + "\n"
    (report_dir / "limitations_facts.md").write_text(text)


def write_ai_disclosure(report_dir: Path) -> None:
    text = """# AI tool use - facts for the student to confirm and edit

ChatGPT
- Used for: planning the project and writing prompts for the coding assistants.
- Produced: plans and prompts.
- Share of work: STUDENT TO CONFIRM.

OpenAI Codex CLI
- Used for: early application code (Streamlit app, detection, incidents, storage).
- Produced: application code and tests.
- Share of work: STUDENT TO CONFIRM.

Claude Code
- Used for: the ML pipeline in `weapon_detection/ml/`, the ML tests, the notebook
  builders, `predict_app.py`, this evidence pack, and README/TASKS updates.
- Produced: code, tables, figures and factual notes (no conclusions).
- Share of work: STUDENT TO CONFIRM.

- The analysis, takeaways, error explanations, discussion and conclusions in the report
  are to be written by the student.
- STUDENT TO CONFIRM: any other tools used (e.g. Colab AI assistance, grammar tools).
- STUDENT TO CONFIRM: how AI output was checked (e.g. tests run, numbers traced to
  `outputs/ml/report_numbers.json`).
"""
    (report_dir / "ai_use_disclosure_facts.md").write_text(text)


def write_presentation_notes(figures, numbers, best_key: str, report_dir: Path) -> None:
    f = lambda prefix: figure_number(figures, prefix)  # noqa: E731
    examples = demo_examples(best_key)
    yolo_path = RESULTS_DIR / "yolov8n_oiv7_pretrained_predictions.csv"
    yolo = pd.read_csv(yolo_path).set_index("image") if yolo_path.exists() else None
    threshold = numbers["chosen_settings"]["values"][best_key]["threshold"]

    lines = [
        "# Presenter notes",
        "",
        "## Slides (6 minutes)",
        "",
        "1. Task and dataset (45 s) - " + f("fig_02"),
        "2. Leakage audit and split v1 -> v2 (75 s) - " + f("fig_06"),
        "3. EDA highlights (45 s) - " + f("fig_01") + ", " + f("fig_03"),
        "4. Models and protocol (60 s) - outputs/ml/chosen_settings.json",
        "5. Results vs baseline (75 s) - outputs/ml/results/final_test_table.md, " + f("fig_10"),
        "6. Errors and limitations (60 s) - "
        + f("fig_12")
        + ", outputs/report/limitations_facts.md",
        "",
        "STUDENT TO WRITE: the message of each slide.",
        "",
        "## Demo (4 minutes)",
        "",
        "Start: `uv run streamlit run predict_app.py` - choose 'Test-split image'.",
        f"Model to select: {NAMES.get(best_key, best_key)} (threshold {threshold}).",
        "",
    ]
    for label, row in examples.items():
        name = Path(row["image"]).name
        lines += [
            f"- {label}: `{name}`",
            f"  - saved score {row['score']:.4f}, true label "
            f"{'firearm' if row['y_true'] else 'no firearm'}, predicted "
            f"{'firearm' if row['y_pred'] else 'no firearm'}",
        ]
        if yolo is not None and row["image"] in yolo.index:
            y = yolo.loc[row["image"]]
            lines.append(
                f"  - YOLOv8n pretrained on the same image: score {y['score']:.4f}, "
                f"predicted {'firearm' if y['y_pred'] else 'no firearm'}"
            )
    lines += [
        "- Then let the audience upload or pick their own image.",
        "",
        "Source: outputs/ml/results/" + f"{best_key}_predictions.csv",
        "",
        "## Likely examiner questions (pointers only; answers by the student)",
        "",
    ]
    questions = [
        (
            "How did you stop the same scene appearing in train and test?",
            [
                line_of("weapon_detection/ml/common.py", "def family_keys"),
                line_of("weapon_detection/ml/prepare_data.py", "def check_no_group_overlap"),
                "outputs/ml/audit.json",
            ],
        ),
        (
            "Why did you change the split after seeing results?",
            [
                line_of("weapon_detection/ml/split_v2.py", "Why: in split v1"),
                "outputs/ml/split_v1/",
                f("fig_06"),
            ],
        ),
        (
            "Was anything fitted on validation or test data?",
            [
                line_of("weapon_detection/ml/classical.py", "def fit_and_choose"),
                line_of("tests/test_ml_classical.py", "def test_scaler_is_fitted_on_train_only"),
            ],
        ),
        (
            "How was the decision threshold chosen?",
            [
                line_of("weapon_detection/ml/classical.py", "def best_threshold"),
                "outputs/ml/chosen_settings.json",
            ],
        ),
        (
            "Why macro-F1 and not accuracy?",
            [
                "outputs/ml/results/final_test_table.md (baseline row)",
                line_of("weapon_detection/ml/classical.py", '"macro_f1": f1_score'),
            ],
        ),
        (
            "Which features does the classical model use?",
            [
                line_of("weapon_detection/ml/features.py", "HANDCRAFTED_NAMES = ["),
                line_of("weapon_detection/ml/features.py", "def hog_features"),
            ],
        ),
        (
            "How is a YOLO detector compared with an image classifier?",
            [line_of("weapon_detection/ml/yolo_eval.py", "def max_firearm_score")],
        ),
        (
            "Which classes count as a firearm?",
            [line_of("weapon_detection/ml/common.py", "FIREARM_CLASS_NAMES = {")],
        ),
        (
            "How do you know the report numbers match the code?",
            [
                line_of("weapon_detection/ml/compare.py", "def check_consistency"),
                "outputs/ml/report_numbers.json",
            ],
        ),
        (
            "Does the GUI use the same features as training?",
            [
                line_of("weapon_detection/ml/features.py", "def features_from_array"),
                line_of(
                    "weapon_detection/ml/predict.py", "handcrafted, hog = features_from_array("
                ),
            ],
        ),
    ]
    for number, (question, pointers) in enumerate(questions, start=1):
        lines += [
            f"{number}. {question}",
            f"   - evidence: {'; '.join(pointers)}",
            "   - STUDENT TO WRITE: answer",
        ]
    (report_dir / "presentation_notes.md").write_text("\n".join(lines) + "\n")


def write_report_pack(numbers: dict, figures, runs, best_key: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_outline(figures, numbers, REPORT_DIR)
    write_limitations(numbers, REPORT_DIR)
    write_ai_disclosure(REPORT_DIR)
    write_presentation_notes(figures, numbers, best_key, REPORT_DIR)
