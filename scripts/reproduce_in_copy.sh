#!/usr/bin/env bash
# Re-run the whole ML pipeline (except Colab training) in a fresh copy of the tracked files
# and time every step. The repository is only read: the dataset ZIP is symlinked, the two
# checkpoints and the archived split v1 results are copied.
#
# Usage:  bash scripts/reproduce_in_copy.sh /tmp/wd_repro
# Writes <copy>/timings.tsv, then copies it to outputs/reproduce/pipeline_runtimes.tsv and
# prints a diff of the copy's final TEST table against outputs/ml/results/final_test_table.csv.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T="${1:?give an empty folder for the copy}"
rm -rf "$T"; mkdir -p "$T"
cd "$REPO" && git ls-files -z --cached --others --exclude-standard | xargs -0 -I{} cp --parents "{}" "$T"/
mkdir -p "$T/datasets/raw" "$T/models" "$T/outputs/ml"
ln -s "$REPO/datasets/raw/Weapon Detection.zip" "$T/datasets/raw/Weapon Detection.zip"
cp "$REPO/models/firearm_yolov8n_best.pt" "$REPO/models/yolov8n-oiv7.pt" "$T/models/"
cp -r "$REPO/outputs/ml/split_v1" "$T/outputs/ml/split_v1"   # archived v1 results (read by compare)
PY="$REPO/.venv/bin/python"
cd "$T"
export YOLO_OFFLINE=true
step() {
  local name="$1"; shift
  local s; s=$(date +%s.%N)
  "$PY" "$@" > "$T/log_$name.txt" 2>&1; local rc=$?
  printf "%s\t%.1f\t%s\n" "$name" "$(echo "$(date +%s.%N) - $s" | bc)" "$rc" >> "$T/timings.tsv"
}
step prepare_data -m weapon_detection.ml.prepare_data
step features -m weapon_detection.ml.features
step eda -m weapon_detection.ml.eda
step classical -m weapon_detection.ml.classical
step yolo_data -m weapon_detection.ml.yolo_data
step yolo_eval_pretrained -m weapon_detection.ml.yolo_eval --checkpoint models/yolov8n-oiv7.pt \
  --name yolov8n_oiv7_pretrained --label "YOLOv8n pretrained (Open Images, not fine-tuned)"
step yolo_eval_finetuned -m weapon_detection.ml.yolo_eval --checkpoint models/firearm_yolov8n_best.pt \
  --name yolov8n_finetuned --label "YOLOv8n fine-tuned (firearm)"
step compare -m weapon_detection.ml.compare
mkdir -p "$REPO/outputs/reproduce"
cp "$T/timings.tsv" "$REPO/outputs/reproduce/pipeline_runtimes.tsv"
diff <(cut -d, -f1-15 "$T/outputs/ml/results/final_test_table.csv") \
     <(cut -d, -f1-15 "$REPO/outputs/ml/results/final_test_table.csv") \
  && echo "final TEST table metrics identical to the repository's (latency column excluded)"
