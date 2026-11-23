"""Step 3: exploratory data analysis figures and one-line factual notes.

Run with:  uv run python -m weapon_detection.ml.eda
Only the class-balance and object-count figures look at all splits (counts only).
Every other figure uses the TRAINING split only, so the test set is not peeked at.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weapon_detection.ml.common import (
    FIGURE_DIR,
    FIREARM_CLASS_NAMES,
    MANIFEST_PATH,
    OUTPUT_DIR,
)
from weapon_detection.ml.features import load_features

SPLITS = ["train", "val", "test"]
CLASS_LABELS = {0: "no firearm", 1: "firearm"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}
PLOTTED_FEATURES = ["mean_brightness", "edge_density", "mean_saturation", "laplacian_sharpness"]


def save(figure: plt.Figure, name: str) -> None:
    """Save a figure as PNG at 150 dpi and close it."""
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / name, dpi=150)
    plt.close(figure)


def plot_class_balance(manifest: pd.DataFrame) -> str:
    counts = pd.crosstab(manifest["split"], manifest["firearm_present"]).loc[SPLITS]
    figure, axis = plt.subplots(figsize=(6, 4))
    x = np.arange(len(SPLITS))
    for offset, label in [(-0.2, 0), (0.2, 1)]:
        bars = axis.bar(
            x + offset,
            counts[label],
            width=0.4,
            label=CLASS_LABELS[label],
            color=CLASS_COLORS[label],
        )
        axis.bar_label(bars)
    axis.set_xticks(x, SPLITS)
    axis.set_ylabel("images")
    axis.set_title("Class balance per new split")
    axis.legend()
    save(figure, "class_balance_per_split.png")
    shares = (counts[1] / counts.sum(axis=1) * 100).round(1)
    return (
        "class_balance_per_split.png: firearm images are "
        + ", ".join(f"{shares[s]}% of {s}" for s in SPLITS)
        + f"; always predicting 'firearm' gives {shares['test']}% accuracy on test."
    )


def plot_object_classes(manifest: pd.DataFrame) -> str:
    rows = []
    for split, classes in zip(manifest["split"], manifest["classes"].fillna("")):
        for name in classes.split(";") if classes else []:
            rows.append((split, name))
    table = pd.crosstab(
        pd.Series([r[1] for r in rows], name="class"), pd.Series([r[0] for r in rows], name="split")
    )[SPLITS]
    table = table.loc[table.sum(axis=1).sort_values().index]
    figure, axis = plt.subplots(figsize=(7, 4))
    table.plot.barh(stacked=True, ax=axis)
    axis.set_xlabel("annotated objects")
    axis.set_title("Annotated objects per class (all splits)")
    save(figure, "object_class_counts.png")
    totals = table.sum(axis=1).sort_values(ascending=False)
    is_firearm = [name.lower() in FIREARM_CLASS_NAMES for name in totals.index]
    firearm_total = int(totals[is_firearm].sum())
    largest_other = totals[[not f for f in is_firearm]].idxmax()
    return (
        f"object_class_counts.png: firearm classes account for {firearm_total} of "
        f"{int(totals.sum())} objects ({firearm_total / totals.sum():.1%}); the largest "
        f"non-firearm class is {largest_other}."
    )


def plot_firearm_box_area(train: pd.DataFrame) -> str:
    areas = []
    for classes, box_areas in zip(train["classes"].fillna(""), train["box_areas"].fillna("")):
        if not classes:
            continue
        for name, area in zip(classes.split(";"), str(box_areas).split(";")):
            if name.lower() in FIREARM_CLASS_NAMES:
                areas.append(float(area))
    areas = np.array(areas)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(areas, bins=50, color=CLASS_COLORS[1])
    axis.set_xlabel("firearm box area / image area")
    axis.set_ylabel("boxes")
    axis.set_title("Firearm box size (train)")
    save(figure, "firearm_box_area_hist.png")
    return (
        f"firearm_box_area_hist.png: the median firearm box covers {np.median(areas):.1%} of "
        f"the image and {np.mean(areas < 0.05):.1%} of {len(areas)} train boxes cover under 5%."
    )


def plot_image_sizes(train: pd.DataFrame) -> str:
    sizes = train["width"].astype(str) + "x" + train["height"].astype(str)
    top = sizes.value_counts().head(10)
    figure, axis = plt.subplots(figsize=(7, 4))
    top[::-1].plot.barh(ax=axis)
    axis.set_xlabel("images")
    axis.set_title(f"Most common image sizes (train, {sizes.nunique()} distinct)")
    save(figure, "image_size_distribution.png")
    return (
        f"image_size_distribution.png: {sizes.nunique()} distinct sizes in train; the most "
        f"common is {top.index[0]} ({top.iloc[0] / len(train):.1%} of train images); "
        "all images are resized to 128x128 before feature extraction."
    )


def plot_feature_distributions(train: pd.DataFrame, features: pd.DataFrame) -> str:
    figure, axes = plt.subplots(2, 2, figsize=(9, 6))
    sentences = []
    for axis, name in zip(axes.flatten(), PLOTTED_FEATURES):
        values = features[name]
        upper = values.quantile(0.99)  # clip the long tail so the plot is readable
        for label in [0, 1]:
            part = values[train["firearm_present"].to_numpy() == label]
            axis.hist(
                part.clip(upper=upper),
                bins=40,
                density=True,
                alpha=0.5,
                label=CLASS_LABELS[label],
                color=CLASS_COLORS[label],
            )
        axis.set_title(name)
        axis.legend()
        neg = values[train["firearm_present"].to_numpy() == 0].median()
        pos = values[train["firearm_present"].to_numpy() == 1].median()
        sentences.append(f"{name} median {pos:.3g} (firearm) vs {neg:.3g} (no firearm)")
    figure.suptitle("Handcrafted feature distributions by class (train, density)")
    save(figure, "feature_distributions.png")
    return "feature_distributions.png: " + "; ".join(sentences) + "."


def main() -> None:
    plt.switch_backend("Agg")  # draw to PNG files, no window needed
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    train_mask = (manifest["split"] == "train").to_numpy()
    train = manifest[train_mask].reset_index(drop=True)

    handcrafted, _, names = load_features(manifest)  # joined by image path
    features = pd.DataFrame(handcrafted, columns=names)
    train_features = features[train_mask].reset_index(drop=True)

    notes = [
        plot_class_balance(manifest),
        plot_object_classes(manifest),
        plot_firearm_box_area(train),
        plot_image_sizes(train),
        plot_feature_distributions(train, train_features),
    ]
    text = "# EDA notes\n\nAll numbers are computed by `weapon_detection/ml/eda.py`. "
    text += "These are factual notes only; the takeaways are written by the student.\n\n"
    text += "\n".join(f"- {note}" for note in notes) + "\n"
    (OUTPUT_DIR / "eda_notes.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
