"""
ARTVISION - Model Evaluation Module
Calculates performance metrics, confusion matrices, classification reports,
and generates publication-quality visualization charts for academic reporting.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CLASS_NAMES = ["Abstract", "Cubism", "Impressionism", "Surrealism"]
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def evaluate_model_on_dataset(
    model: Any,
    test_dir: str,
    reports_dir: str = REPORTS_DIR,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Evaluates a trained Keras model on a directory of labeled test images.
    Saves confusion matrix image and metrics JSON.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    os.makedirs(reports_dir, exist_ok=True)

    y_true = []
    y_pred = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    from src.preprocessing import load_and_preprocess_image

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.exists(class_folder):
            continue

        file_list = [f for f in os.listdir(class_folder) if os.path.splitext(f)[1].lower() in valid_exts]
        for fname in file_list:
            fpath = os.path.join(class_folder, fname)
            try:
                batch, _ = load_and_preprocess_image(fpath)
                preds = model.predict(batch, verbose=0)
                pred_idx = int(np.argmax(preds[0]))
                y_true.append(class_idx)
                y_pred.append(pred_idx)
            except Exception as e:
                print(f"Warning: skipped {fpath}: {e}")

    if not y_true:
        print(f"[Warning] No test images found in {test_dir} to evaluate.")
        return {}

    # Calculate metrics
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("=" * 50)
    print(report_str)

    # Save metrics JSON
    metrics_path = os.path.join(reports_dir, "classification_report.json")
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved classification report to: {metrics_path}")

    # Plot Confusion Matrix
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, class_names, cm_path)

    return report_dict


def plot_and_save_confusion_matrix(cm, class_names: List[str], save_path: str):
    """Render and save styled confusion matrix heatmap."""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Artwork Classification - Confusion Matrix",
        ylabel="True Artwork Style",
        xlabel="Predicted Artwork Style"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate values inside matrix
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, i] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix chart to: {save_path}")


def load_saved_metrics(reports_dir: str = REPORTS_DIR) -> Optional[Dict[str, Any]]:
    """Loads existing classification report JSON if previously generated."""
    metrics_path = os.path.join(reports_dir, "classification_report.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    from src.train_model import DEFAULT_MODEL_PATH
    import tensorflow as tf

    test_path = os.path.join(PROJECT_ROOT, "dataset", "test")
    if os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(test_path):
        print(f"Loading model from {DEFAULT_MODEL_PATH} for evaluation...")
        model = tf.keras.models.load_model(DEFAULT_MODEL_PATH)
        evaluate_model_on_dataset(model, test_path)
    else:
        print("Model or test dataset not found. Run training first.")
