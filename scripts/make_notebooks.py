"""Write notebooks 01, 02 and 04 as plain JSON (no nbformat needed).

Run with:  uv run python scripts/make_notebooks.py
Execute with:
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_eda.ipynb
The notebooks only CALL the modules in weapon_detection/ml/; no logic is duplicated.
Notebook 03 (Colab) is written by scripts/make_colab_notebook.py.
"""

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SEED_CELL = """# Fixed seed for everything random in this notebook (the modules use SEED = 42 too).
import random
import time

import numpy as np
import pandas as pd
from IPython.display import Image, display

from weapon_detection.ml.common import OUTPUT_DIR, SEED

random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.width", 200)
notebook_start = time.perf_counter()
print("SEED =", SEED)"""

TIMER_CELL = """# Measured runtime of this notebook on the machine that executed it.
print(f"Notebook runtime: {time.perf_counter() - notebook_start:.0f} seconds")"""


def how_to(title: str, commands: list[str], note: str) -> str:
    lines = [
        f"# {title}",
        "",
        "## How to reproduce",
        "",
        "From the repository root (Python 3.11, uv installed):",
        "",
        "```sh",
        "uv sync --locked",
        *commands,
        "```",
        "",
        note,
        "The measured runtime is printed in the last cell.",
    ]
    return "\n".join(lines)


NOTEBOOKS = {
    "01_data_and_eda.ipynb": [
        (
            how_to(
                "01 - Data loading, leakage audit, split and EDA",
                [
                    "uv run jupyter nbconvert --to notebook --execute --inplace "
                    "notebooks/01_data_and_eda.ipynb"
                ],
                "Needs `datasets/raw/Weapon Detection.zip`. The first run extracts the ZIP and "
                "computes features (a few minutes on a laptop CPU); later runs use the cache.",
            ),
            None,
        ),
        (
            "**Seed.** Everything random uses seed 42, so every run gives the same split and "
            "the same models.",
            SEED_CELL,
        ),
        (
            "**Step 1 - Load, label and audit the data.** `prepare_data.main()` checks the ZIP, "
            "extracts it safely, reads every YOLO label, marks an image as *firearm* if it has a "
            "Gun or Pistol box, finds duplicates / near duplicates / video-frame families, and "
            "builds the group-aware split. We split BEFORE any model is trained so the test set "
            "stays unseen.",
            "from weapon_detection.ml import prepare_data\n\nprepare_data.main()",
        ),
        (
            "**Step 2 - Look at the manifest.** One row per image: path, split, group, label. "
            "Models never see the path, group or split as inputs.",
            "manifest = pd.read_csv(OUTPUT_DIR / 'manifest.csv')\n"
            "display(manifest.head())\n"
            "pd.crosstab(manifest['split'], manifest['firearm_present'], margins=True)",
        ),
        (
            "**Step 3 - Pre-processing: features.** Each image is resized to 128x128; we compute "
            "10 interpretable numbers (brightness, edges, colour, ...) and a HOG vector. These are "
            "per-image calculations, so nothing is *fitted* here (scaling is fitted later on the "
            "training split only).",
            "from weapon_detection.ml import features\n\n"
            "features.main()\n"
            "handcrafted, hog, names = features.load_features(manifest)\n"
            "print('handcrafted', handcrafted.shape, 'hog', hog.shape)\n"
            "print(names)",
        ),
        (
            "**Step 4 - Exploratory data analysis.** Figures use the TRAINING split only "
            "(except the class counts), so we do not peek at the test set. The notes are facts; "
            "the takeaways are written by me in the report.",
            "from weapon_detection.ml import eda\n\n"
            "eda.main()\n"
            "for name in ['class_balance_per_split', 'object_class_counts', "
            "'firearm_box_area_hist',\n"
            "             'image_size_distribution', 'feature_distributions']:\n"
            "    display(Image(filename=str(OUTPUT_DIR / 'figures' / f'{name}.png'), width=650))",
        ),
        (
            "**Step 5 - Leakage audit numbers.** Where do the negatives of each split come from? "
            "Split v1 vs split v2 (v2 was introduced after preliminary v1 results; see report).",
            "import json\n\n"
            "audit = json.loads((OUTPUT_DIR / 'audit.json').read_text())\n"
            "for version in ['negative_sources_v1', 'negative_sources_v2']:\n"
            "    print(version)\n"
            "    display(pd.DataFrame(audit['split_v2'][version]).T)",
        ),
        ("**Runtime.**", TIMER_CELL),
    ],
    "02_classical_models.ipynb": [
        (
            how_to(
                "02 - Classical models: baseline, Logistic Regression, Random Forest",
                [
                    "uv run python -m weapon_detection.ml.classical   # full run, test used once",
                    "uv run jupyter nbconvert --to notebook --execute --inplace "
                    "notebooks/02_classical_models.ipynb",
                ],
                "The full training + single test evaluation was done by "
                "`weapon_detection/ml/classical.py`. This notebook re-runs the TUNING on "
                "train/validation only to show how settings were chosen, and then LOADS the "
                "saved test results - it does not evaluate the test set again.",
            ),
            None,
        ),
        ("**Seed.**", SEED_CELL),
        (
            "**Step 1 - Load features and labels.** Features are joined to the manifest by image "
            "path, never by row order.",
            "from weapon_detection.ml.features import load_features\n\n"
            "manifest = pd.read_csv(OUTPUT_DIR / 'manifest.csv')\n"
            "handcrafted, hog, names = load_features(manifest)\n"
            "X = {'handcrafted': handcrafted, 'handcrafted_hog': np.hstack([handcrafted, hog])}\n"
            "y = manifest['firearm_present'].to_numpy()\n"
            "split = manifest['split'].to_numpy()\n"
            "print({s: int((split == s).sum()) for s in ['train', 'val', 'test']})",
        ),
        (
            "**Step 2 - Pre-processing inside a Pipeline.** The StandardScaler is a step of the "
            "pipeline, so `fit()` on the training rows is the only place it learns a mean and "
            "standard deviation. Trees do not need scaling.",
            "from weapon_detection.ml.classical import make_pipeline\n\n"
            "print(make_pipeline('baseline', {}))\n"
            "print(make_pipeline('logreg', {'C': 0.01}))\n"
            "print(make_pipeline('random_forest', {'max_depth': 10, 'min_samples_leaf': 1}))",
        ),
        (
            "**Step 3 - Training and tuning on train/validation only.** For each grid value we fit "
            "on train and score macro-F1 on validation. Then we pick the decision threshold on "
            "the validation scores. (The Random Forest on HOG takes ~2 minutes and is tuned in "
            "classical.py; here we show the three faster models.) We check that we get the same "
            "settings that were saved before the test evaluation.",
            "import json\n\n"
            "from weapon_detection.ml.classical import fit_and_choose\n\n"
            "saved = json.loads((OUTPUT_DIR / 'chosen_settings.json').read_text())\n"
            "for model, feature_set in [('logreg', 'handcrafted'), ('random_forest', "
            "'handcrafted'),\n"
            "                           ('logreg', 'handcrafted_hog')]:\n"
            "    pipeline, chosen = fit_and_choose(model, X[feature_set], y, split)\n"
            "    key = f'{model}_{feature_set}'\n"
            "    display(pd.DataFrame(chosen['tuning_log']))\n"
            "    print(key, 'params', chosen['params'], 'threshold', chosen['threshold'])\n"
            "    assert chosen['params'] == saved[key]['params']\n"
            "    assert chosen['threshold'] == saved[key]['threshold']",
        ),
        (
            "**Step 4 - Test results (loaded, not recomputed).** The test split was evaluated "
            "exactly once by classical.py after the settings above were saved. The baseline shows "
            "accuracy next to macro-F1: it never predicts 'no firearm'.",
            "metrics = pd.read_csv(OUTPUT_DIR / 'results' / 'classical_metrics.csv')\n"
            "columns = ['model', 'feature_set', 'split', 'threshold', 'precision', 'recall',\n"
            "           'f1', 'accuracy', 'macro_f1', 'neg_recall', 'roc_auc']\n"
            "metrics[columns]",
        ),
        (
            "**Step 5 - Confusion matrices, ROC and PR curves on test.**",
            "names = ['confusion_logreg_handcrafted_hog_test',\n"
            "         'confusion_random_forest_handcrafted_hog_test', 'roc_pr_test']\n"
            "for name in names:\n"
            "    display(Image(filename=str(OUTPUT_DIR / 'figures' / f'{name}.png'), width=650))",
        ),
        (
            "**Step 6 - What the models use.** Logistic Regression coefficients (scaled features) "
            "and Random Forest impurity importance.",
            "display(pd.read_csv(OUTPUT_DIR / 'results' / 'lr_coefficients_handcrafted_hog.csv'))\n"
            "pd.read_csv(OUTPUT_DIR / 'results' / 'rf_importance.csv')",
        ),
        (
            "**Step 7 - Error analysis data.** The most confident wrong predictions on test. The "
            "`my_explanation` column is for me to fill in.",
            "display(pd.read_csv(OUTPUT_DIR / 'error_analysis.csv'))\n"
            "Image(filename=str(OUTPUT_DIR / 'figures' / 'error_analysis_grid.png'), width=900)",
        ),
        ("**Runtime.**", TIMER_CELL),
    ],
    "04_comparison.ipynb": [
        (
            how_to(
                "04 - Final comparison of all models",
                [
                    "uv run python -m weapon_detection.ml.compare",
                    "uv run jupyter nbconvert --to notebook --execute --inplace "
                    "notebooks/04_comparison.ipynb",
                ],
                "compare.py only reads saved results (it re-checks every metric against the "
                "saved predictions). The fine-tuned YOLO row appears automatically after "
                "`yolo_eval.py` has been run on its checkpoint.",
            ),
            None,
        ),
        # Notebook 04 links figures instead of embedding them, so it does not import Image.
        ("**Seed.**", SEED_CELL.replace("import Image, display", "import display")),
        (
            "**Step 1 - Regenerate tables, figures and report numbers.** This also checks that "
            "every metric in the table can be recomputed from the saved prediction files.",
            "from weapon_detection.ml import compare\n\ncompare.main()",
        ),
        (
            "**Step 2 - Image-level results on TEST and VALIDATION.** Firearm is the positive "
            "class; `neg_*` columns are the no-firearm class.",
            "from IPython.display import Markdown\n\n"
            "results = OUTPUT_DIR / 'results'\n"
            "display(Markdown((results / 'final_test_table.md').read_text()))\n"
            "display(Markdown((results / 'final_val_table.md').read_text()))",
        ),
        (
            "**Step 3 - Detection metrics are kept separate** (mAP is a different kind of number "
            "from image-level F1).",
            "Markdown((OUTPUT_DIR / 'results' / 'detection_metrics.md').read_text())",
        ),
        (
            "**Step 4 - Comparison figures.** The figures are linked, not embedded, to keep "
            "this notebook small. Copies are in `report_assets/figures/` "
            "(`uv run python scripts/build_report_assets.py`); originals in "
            "`outputs/ml/figures/`.",
            "from pathlib import Path\n\n"
            "assets = Path('..') / 'report_assets' / 'figures'  # relative to notebooks/\n"
            "names = [p.name for p in sorted((OUTPUT_DIR / 'figures').glob('fig_*.png'))]\n"
            "display(Markdown('\\n\\n'.join(f'**{n}**\\n\\n![{n}]({(assets / n).as_posix()})' "
            "for n in names)))",
        ),
        (
            "**Step 5 - Course requirement checklist** (strict; generated from the files).",
            "Markdown((OUTPUT_DIR / 'requirements_check.md').read_text())",
        ),
        ("**Runtime.**", TIMER_CELL),
    ],
}


def to_notebook(cells: list[tuple[str, str | None]]) -> dict:
    out = []
    for markdown, code in cells:
        out.append({"cell_type": "markdown", "metadata": {}, "source": markdown})
        if code is not None:
            out.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": code,
                }
            )
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (NOTEBOOK_DIR / name).write_text(json.dumps(to_notebook(cells), indent=1))
        print("Wrote", NOTEBOOK_DIR / name)


if __name__ == "__main__":
    main()
