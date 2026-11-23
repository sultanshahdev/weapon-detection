# Image-level results on the TEST split

Firearm = positive class. neg_* = no-firearm class. Threshold chosen on validation. ms/image: classical = model only on precomputed features; YOLO = read + inference.

| model | family | main_model | precision | recall | f1 | accuracy | balanced_accuracy | macro_f1 | neg_precision | neg_recall | neg_f1 | roc_auc | pr_auc | threshold | predict_ms_per_image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | trivial (DummyClassifier) |  | 0.9023 | 1.0000 | 0.9486 | 0.9023 | 0.5000 | 0.4743 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.9023 |  | 0.0006 |
| Logistic Regression (handcrafted) | linear model |  | 0.9216 | 0.9194 | 0.9205 | 0.8567 | 0.5986 | 0.5976 | 0.2717 | 0.2778 | 0.2747 | 0.6371 | 0.9382 | 0.3200 | 0.0013 |
| Random Forest (handcrafted) | tree ensemble |  | 0.9290 | 0.9290 | 0.9290 | 0.8719 | 0.6367 | 0.6367 | 0.3444 | 0.3444 | 0.3444 | 0.7257 | 0.9494 | 0.4900 | 0.2478 |
| Logistic Regression (handcrafted + HOG) | linear model | yes | 0.9335 | 0.8953 | 0.9140 | 0.8480 | 0.6532 | 0.6299 | 0.2984 | 0.4111 | 0.3458 | 0.7449 | 0.9583 | 0.4700 | 0.0105 |
| Random Forest (handcrafted + HOG) | tree ensemble | yes | 0.9344 | 0.9434 | 0.9389 | 0.8893 | 0.6662 | 0.6729 | 0.4268 | 0.3889 | 0.4070 | 0.7893 | 0.9677 | 0.7600 | 0.2611 |
| YOLOv8n pretrained (Open Images, not fine-tuned) | CNN object detector |  | 0.9781 | 0.4308 | 0.5982 | 0.4777 | 0.6710 | 0.4262 | 0.1477 | 0.9111 | 0.2543 | 0.6661 | 0.9343 | 0.0100 | 107.7535 |
| YOLOv8n fine-tuned (firearm) | CNN object detector | yes | 0.9647 | 0.9543 | 0.9595 | 0.9273 | 0.8160 | 0.8025 | 0.6162 | 0.6778 | 0.6455 | 0.9211 | 0.9887 | 0.1200 | 146.6271 |
