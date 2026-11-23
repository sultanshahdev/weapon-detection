"""Write notebooks/03_yolo_colab.ipynb as plain JSON (no nbformat needed).

Run with:  uv run python scripts/make_colab_notebook.py
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "03_yolo_colab.ipynb"

# (markdown explanation, code) pairs, one per step.
CELLS = [
    (
        "# 03 - Fine-tune YOLOv8n on the firearm dataset (Colab, GPU)\n\n"
        "## How to reproduce\n\n"
        "1. Locally: `uv run python -m weapon_detection.ml.yolo_data` creates "
        "`outputs/yolo_colab/yolo_firearm.zip`.\n"
        "2. Upload that zip to the top level of Google Drive (`MyDrive`).\n"
        "3. In Colab: Runtime -> Change runtime type -> **T4 GPU**, then Run all.\n"
        "4. Download `MyDrive/yolo_results/best.pt` to `models/firearm_yolov8n_best.pt`.\n\n"
        "Seed 42 is passed to the trainer (`seed=42, deterministic=True`). Runtime depends "
        "on the GPU; Ultralytics prints the time per epoch.",
        None,
    ),
    (
        "**1. Mount Google Drive** so the notebook can read the zip and save results.",
        "from google.colab import drive\n\ndrive.mount('/content/drive')",
    ),
    (
        "**2. Unzip the dataset** to the fast local disk of the Colab machine.",
        "!unzip -q -o /content/drive/MyDrive/yolo_firearm.zip -d /content\n"
        "!ls /content/yolo_firearm",
    ),
    (
        "**3. Install Ultralytics** (the YOLO library).",
        "!pip install -q ultralytics",
    ),
    (
        "**4. Fix the dataset path in data.yaml.** It still points to the folder on my "
        "laptop; YOLO needs the Colab folder.",
        "import yaml\n\n"
        "path = '/content/yolo_firearm/data.yaml'\n"
        "data = yaml.safe_load(open(path))\n"
        "data['path'] = '/content/yolo_firearm'\n"
        "yaml.safe_dump(data, open(path, 'w'), sort_keys=False)\n"
        "print(open(path).read())",
    ),
    (
        "**5. Check the GPU.** Training on CPU would take many hours.",
        "import torch\n\n"
        "print('CUDA available:', torch.cuda.is_available())\n"
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')",
    ),
    (
        "**6. Train.** Start from COCO-pretrained `yolov8n.pt` (transfer learning) with the "
        "same settings as `weapon_detection/ml/yolo_train.py`.",
        "from ultralytics import YOLO\n\n"
        "args = dict(data='/content/yolo_firearm/data.yaml', epochs=30, imgsz=640, batch=16,\n"
        "            patience=10, seed=42, deterministic=True, workers=2, plots=True,\n"
        "            device=0, project='/content/yolo_runs', name='firearm_yolov8n')\n"
        "print(args)\n"
        "model = YOLO('yolov8n.pt')\n"
        "model.train(**args)",
    ),
    (
        "**7. Copy the results back to Drive** (Colab's disk is deleted when the session ends).",
        "import os\n"
        "import shutil\n\n"
        "run = '/content/yolo_runs/firearm_yolov8n'\n"
        "out = '/content/drive/MyDrive/yolo_results'\n"
        "os.makedirs(out, exist_ok=True)\n"
        "for name in ['weights/best.pt', 'results.csv', 'results.png', 'confusion_matrix.png',\n"
        "             'args.yaml']:\n"
        "    shutil.copy(os.path.join(run, name), out)\n"
        "    print('copied', name)",
    ),
    (
        "**8. Show the training curves and the validation confusion matrix.**",
        "import pandas as pd\n"
        "from IPython.display import Image, display\n\n"
        "display(Image(run + '/results.png'))\n"
        "display(Image(run + '/confusion_matrix.png'))\n"
        "pd.read_csv(run + '/results.csv').tail()",
    ),
]


def main() -> None:
    cells = []
    for markdown, code in CELLS:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": markdown})
        if code is not None:
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": code,
                }
            )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
