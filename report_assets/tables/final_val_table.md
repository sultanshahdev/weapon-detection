# Image-level results on the VAL split

Firearm = positive class. neg_* = no-firearm class. Threshold chosen on validation. ms/image: classical = model only on precomputed features; YOLO = read + inference.

| model | family | main_model | precision | recall | f1 | accuracy | balanced_accuracy | macro_f1 | neg_precision | neg_recall | neg_f1 | roc_auc | pr_auc | threshold | predict_ms_per_image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | trivial (DummyClassifier) |  | 0.9023 | 1.0000 | 0.9486 | 0.9023 | 0.5000 | 0.4743 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.9023 |  | 0.0006 |
| Logistic Regression (handcrafted) | linear model |  | 0.9300 | 0.8797 | 0.9041 | 0.8317 | 0.6343 | 0.6076 | 0.2593 | 0.3889 | 0.3111 | 0.6621 | 0.9391 | 0.3200 | 0.0013 |
| Random Forest (handcrafted) | tree ensemble |  | 0.9362 | 0.9182 | 0.9271 | 0.8697 | 0.6702 | 0.6574 | 0.3585 | 0.4222 | 0.3878 | 0.7563 | 0.9611 | 0.4900 | 0.1478 |
| Logistic Regression (handcrafted + HOG) | linear model | yes | 0.9409 | 0.9194 | 0.9300 | 0.8751 | 0.6930 | 0.6761 | 0.3853 | 0.4667 | 0.4221 | 0.7732 | 0.9631 | 0.4700 | 0.0103 |
| Random Forest (handcrafted + HOG) | tree ensemble | yes | 0.9366 | 0.9422 | 0.9394 | 0.8903 | 0.6767 | 0.6811 | 0.4353 | 0.4111 | 0.4229 | 0.8021 | 0.9683 | 0.7600 | 0.1996 |
| YOLOv8n pretrained (Open Images, not fine-tuned) | CNN object detector |  | 0.9920 | 0.4465 | 0.6158 | 0.4973 | 0.7066 | 0.4445 | 0.1590 | 0.9667 | 0.2732 | 0.7039 | 0.9416 | 0.0100 | 100.6229 |
| YOLOv8n fine-tuned (firearm) | CNN object detector | yes | 0.9723 | 0.9711 | 0.9717 | 0.9490 | 0.8578 | 0.8560 | 0.7363 | 0.7444 | 0.7403 | 0.9173 | 0.9862 | 0.1200 | 149.7863 |
