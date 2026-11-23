# Notice: data attribution and licences

This file covers the datasets only. **TODO: choose a licence for the code in this
repository.** No code licence has been chosen yet, so no licence is granted for the code.

## Training and evaluation data: Roboflow "Weapon Detection-1", version 4

Used for every reported result (training, validation and test). Attribution, copied
exactly from `README.dataset.txt` inside the downloaded export:

```text
# Weapon Detection-1 > OAWD_6_Classes-sohas-
https://universe.roboflow.com/weapon-detection-oss9n/weapon-detection-1-f9p7t

Provided by a Roboflow user
License: CC BY 4.0
```

- Version and export: "Weapon Detection-1 - v4 OAWD_6_Classes-sohas-" (`README.roboflow.txt`),
  YOLOv8 format, <https://universe.roboflow.com/weapon-detection-oss9n/weapon-detection-1-f9p7t/dataset/4>.
- Licence as stated in the dataset's files (`README.dataset.txt` and `data.yaml`): CC BY 4.0,
  <https://creativecommons.org/licenses/by/4.0/>.
- TODO: verify the project title on the Roboflow page. It may be shown there as "Pose
  Integrated Firearm Detection". The downloaded files only use the name above, and the page
  could not be opened automatically (HTTP 403 on 2026-11-23).
- The images themselves are not redistributed in this repository.

## Demo footage only: Mendeley "Action recognition and object detection dataset for firearm-related actions"

Used only by `video_demo_app.py` for a non-commercial coursework demonstration. It is not
used for training, validation, threshold selection or any reported result, and no video or
frame from it is committed.

- Authors: Jesus Ruiz-Santaquiteria, Juan D. Munoz, Francisco J. Maigler, Oscar Deniz,
  Gloria Bueno. Mendeley Data, version 2, DOI 10.17632/bbzpxhd22j.2,
  <https://data.mendeley.com/datasets/bbzpxhd22j/2>.
- Licence on the dataset page: "CC BY NC 3.0". The page also states: "The dataset is
  licensed under CC BY NC 3.0. However, we do allow commercial use, but only if prior
  permission is obtained."
- Related article: "Firearm-related action recognition and object detection dataset for
  video surveillance systems", Data in Brief 52 (2024) 110030, DOI 10.1016/j.dib.2024.110030
  (the article is published under CC BY 4.0).

## Pretrained weights

`models/yolov8n-oiv7.pt` is the Ultralytics YOLOv8n Open Images V7 checkpoint and the
fine-tuned model starts from Ultralytics `yolov8n.pt`. Ultralytics models are distributed
under Ultralytics' own terms (<https://www.ultralytics.com/license>). Weights are not
committed.
