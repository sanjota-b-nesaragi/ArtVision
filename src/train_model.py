"""
ARTVISION - Model Training Script
Implements Deep Learning Transfer Learning using MobileNetV2 (or EfficientNetB0)
for classifying artwork into four distinct styles:
1. Impressionism
2. Cubism
3. Abstract
4. Surrealism

Generates real evaluation metrics, confusion matrix, accuracy and loss curves.
"""

import os
import sys
import json
import argparse
from typing import Tuple, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ["Abstract", "Cubism", "Impressionism", "Surrealism"]

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "artwork_classifier.keras")


def check_dataset_status(dataset_root: str = DATASET_DIR) -> Dict[str, Any]:
    """
    Inspect the dataset directory to count images per split and class.
    """
    status = {"splits": {}, "total_images": 0, "is_ready": False}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for split in ["train", "validation", "test"]:
        split_path = os.path.join(dataset_root, split)
        status["splits"][split] = {}
        split_total = 0
        if os.path.exists(split_path):
            for cls in CLASS_NAMES:
                cls_path = os.path.join(split_path, cls)
                if os.path.exists(cls_path):
                    count = len([
                        f for f in os.listdir(cls_path)
                        if os.path.splitext(f)[1].lower() in valid_exts
                    ])
                else:
                    count = 0
                status["splits"][split][cls] = count
                split_total += count
        status["splits"][split]["total"] = split_total
        status["total_images"] += split_total

    # Ready if training split has at least 1 image per class
    train_counts = [status["splits"]["train"].get(cls, 0) for cls in CLASS_NAMES]
    if all(c > 0 for c in train_counts):
        status["is_ready"] = True

    return status


def build_transfer_learning_model(
    model_architecture: str = "mobilenet_v2",
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3)
):
    """
    Constructs a Transfer Learning model with MobileNetV2 or EfficientNetB0 backbone.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    if model_architecture.lower() == "efficientnet_b0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        preprocess_layer = layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)
    else:  # Default MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        preprocess_layer = layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)

    # Freeze base model layers initially
    base_model.trainable = False

    # Classification head
    inputs = layers.Input(shape=input_shape, name="artwork_input")
    x = preprocess_layer(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_feature_1"
    )(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_feature_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="artwork_predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"artvision_{model_architecture}")
    return model, base_model


def train(
    dataset_dir: str = DATASET_DIR,
    model_architecture: str = "mobilenet_v2",
    epochs: int = 15,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 1e-4,
    fine_tune: bool = True,
    fine_tune_epochs: int = 5,
    save_path: str = DEFAULT_MODEL_PATH
):
    """
    Execute end-to-end model training, validation, and evaluation.
    """
    import tensorflow as tf
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("ARTVISION - ARTWORK STYLE CLASSIFICATION MODEL TRAINING")
    print("=" * 60)

    # Check dataset
    status = check_dataset_status(dataset_dir)
    print(f"Dataset summary at '{dataset_dir}':")
    for split, info in status["splits"].items():
        print(f"  [{split.upper()}] Total: {info.get('total', 0)} images")
        for cls in CLASS_NAMES:
            print(f"     - {cls}: {info.get(cls, 0)}")

    if not status["is_ready"]:
        print("\n[ERROR] Training dataset does not have images in all 4 categories!")
        print("Please run `python src/dataset_setup.py` or populate `dataset/train/` first.")
        return False

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")
    test_dir = os.path.join(dataset_dir, "test")

    # Load datasets using Keras utility
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    # Use data augmentation on training set
    from src.preprocessing import get_data_augmentation_pipeline
    aug_pipeline = get_data_augmentation_pipeline()

    if aug_pipeline:
        train_ds = train_ds.map(lambda x, y: (aug_pipeline(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build model
    print(f"\nBuilding model architecture: {model_architecture}...")
    model, base_model = build_transfer_learning_model(model_architecture)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\nStarting Phase 1: Feature Extraction ({epochs} epochs)...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Optional Fine-Tuning Phase
    if fine_tune and fine_tune_epochs > 0:
        print(f"\nStarting Phase 2: Fine-Tuning Top Layers ({fine_tune_epochs} epochs)...")
        base_model.trainable = True
        # Freeze all layers except the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=len(history_phase1.history["loss"]),
            callbacks=callbacks
        )

        # Merge histories
        combined_history = {}
        for k in history_phase1.history:
            combined_history[k] = history_phase1.history[k] + history_phase2.history[k]
    else:
        combined_history = history_phase1.history

    # Save final model
    model.save(save_path)
    print(f"\nModel successfully saved to: {save_path}")

    # Generate and save Accuracy & Loss curves
    save_training_plots(combined_history, REPORTS_DIR)

    # Evaluate on Test Set
    if os.path.exists(test_dir):
        print("\nEvaluating model on Test Set...")
        evaluate_test_set(model, test_dir, batch_size, REPORTS_DIR)

    return True


def save_training_plots(history: Dict[str, list], reports_dir: str):
    """Save accuracy and loss graphs to reports folder."""
    import matplotlib.pyplot as plt

    epochs_range = range(1, len(history["accuracy"]) + 1)

    # Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history["accuracy"], "o-", label="Training Accuracy", color="#2563EB", linewidth=2)
    plt.plot(epochs_range, history["val_accuracy"], "s--", label="Validation Accuracy", color="#10B981", linewidth=2)
    plt.title("Artwork Classification - Training & Validation Accuracy", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    acc_path = os.path.join(reports_dir, "accuracy.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy plot to: {acc_path}")

    # Loss Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history["loss"], "o-", label="Training Loss", color="#DC2626", linewidth=2)
    plt.plot(epochs_range, history["val_loss"], "s--", label="Validation Loss", color="#F59E0B", linewidth=2)
    plt.title("Artwork Classification - Training & Validation Loss", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Categorical Crossentropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    loss_path = os.path.join(reports_dir, "loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to: {loss_path}")


def evaluate_test_set(model, test_dir: str, batch_size: int, reports_dir: str):
    """Compute confusion matrix and classification report on test set."""
    from src.evaluation import evaluate_model_on_dataset
    evaluate_model_on_dataset(model, test_dir, reports_dir, CLASS_NAMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ARTVISION Artwork Classifier")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="Path to dataset root folder")
    parser.add_argument("--model-type", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune", action="store_true", default=False)
    parser.add_argument("--check-dataset", action="store_true", help="Inspect dataset status without training")

    args = parser.parse_args()

    if args.check_dataset:
        st = check_dataset_status(args.dataset_dir)
        print("Dataset Status:")
        print(json.dumps(st, indent=2))
    else:
        train(
            dataset_dir=args.dataset_dir,
            model_architecture=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            fine_tune=args.fine_tune
        )
