# Demo footage diagnostics (Mendeley videos; demo only, not an evaluation)

## D1. Box size

Columns: width % = box width / image width x 100; height % = box height / image height x 100; area % = box area / image area x 100; pN = N-th percentile over all boxes. Mendeley: every box in every Handgun and Machine_Gun label.json. Roboflow: every class-0 (firearm) box in outputs/yolo_firearm/labels/train.

| source | boxes | width % p10 | width % p50 | width % p90 | height % p10 | height % p50 | height % p90 | area % p10 | area % p50 | area % p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mendeley Handgun (GT) | 20545 | 2.92 | 4.17 | 6.38 | 5.19 | 6.85 | 10.05 | 0.17 | 0.28 | 0.62 |
| Mendeley Machine_Gun (GT) | 22240 | 4.22 | 9.69 | 18.65 | 6.96 | 16.61 | 33.71 | 0.58 | 1.45 | 3.97 |
| Roboflow train split (firearm) | 2838 | 9.96 | 36.47 | 94.95 | 13.74 | 36.73 | 84.75 | 1.78 | 12.89 | 71.03 |

## D2. Inference size 640 vs 960 (three demo videos, confidence 0.25)

Checkpoint firearm_yolov8n_best.pt, CPU, every frame. Columns: frames = decoded frames; gt_frames = frames with a dataset box; frames_with_model_box = frames with at least one model box; gt_frames_matched_iou50 = GT frames where a model box overlaps a GT box with IoU >= 0.5. The app uses 640 (the training size); these numbers are not stored in the app cache.

| video | imgsz | frames | gt_frames | frames_with_model_box | gt_frames_matched_iou50 |
|---|---|---|---|---|---|
| Handgun/PCH1_C1_P4_V1_HB_2 | 640 | 125 | 78 | 99 | 0 |
| Machine_Gun/PCW4_C1_P4_V1_HB_1 | 640 | 175 | 81 | 139 | 0 |
| No_Gun/N11_C2_P5_V1_HB_2 | 640 | 300 | 0 | 29 | 0 |
| Handgun/PCH1_C1_P4_V1_HB_2 | 960 | 125 | 78 | 115 | 4 |
| Machine_Gun/PCW4_C1_P4_V1_HB_1 | 960 | 175 | 81 | 158 | 0 |
| No_Gun/N11_C2_P5_V1_HB_2 | 960 | 300 | 0 | 21 | 0 |
