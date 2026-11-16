"""
ARTVISION - Prediction Inference Engine
Handles loading of trained Keras transfer learning models, image preprocessing,
multi-class softmax probability calculation, and database logging.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_preprocess_image, ImageValidationError
from src.database import log_classification

CLASS_NAMES = ["Abstract", "Cubism", "Impressionism", "Surrealism"]
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "artwork_classifier.keras")

# Global cached model to prevent reloading on every Streamlit rerun
_CACHED_MODEL = None


class ModelNotTrainedError(Exception):
    """Raised when inference is attempted but the model file has not yet been trained or found."""
    pass


def is_model_available(model_path: str = MODEL_PATH) -> bool:
    """Check if a trained Keras model file exists on disk."""
    return os.path.exists(model_path) and os.path.getsize(model_path) > 1024


def load_classifier_model(model_path: str = MODEL_PATH, force_reload: bool = False):
    """
    Loads and caches the trained Keras transfer learning model.
    """
    global _CACHED_MODEL

    if not is_model_available(model_path):
        raise ModelNotTrainedError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by running `python src/train_model.py`."
        )

    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL

    try:
        import tensorflow as tf
        _CACHED_MODEL = tf.keras.models.load_model(model_path)
        return _CACHED_MODEL
    except Exception as e:
        raise RuntimeError(f"Failed to load trained model from '{model_path}': {e}")


def predict_artwork_style(
    image_input: Any,
    filename: str = "uploaded_artwork.jpg",
    model_path: str = MODEL_PATH,
    log_to_db: bool = True
) -> Dict[str, Any]:
    """
    Perform deep learning inference to classify an artwork image into one of 4 styles:
    Impressionism, Cubism, Abstract, Surrealism.

    Returns:
        Dictionary containing:
        - 'predicted_style': str
        - 'confidence': float (0-100%)
        - 'probabilities': Dict[str, float] (percentage per class)
        - 'raw_scores': List[float]
        - 'is_trained_model': bool (True if from actual saved model)
        - 'log_id': int (database record id)
    """
    # 1. Preprocess image
    batch_array, pil_img = load_and_preprocess_image(image_input)

    # 2. Check if model exists
    if not is_model_available(model_path):
        raise ModelNotTrainedError(
            f"The classification model has not yet been trained (missing '{model_path}'). "
            "Follow the instructions on the Home page or run `python src/train_model.py` to train it."
        )

    # 3. Run Inference
    model = load_classifier_model(model_path)
    predictions = model.predict(batch_array, verbose=0)[0]

    # Map predictions to class probabilities
    probabilities = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        probabilities[class_name] = round(float(predictions[idx]) * 100.0, 2)

    # Top class
    import numpy as np
    top_idx = int(np.argmax(predictions))
    predicted_style = CLASS_NAMES[top_idx]
    confidence = round(float(predictions[top_idx]) * 100.0, 2)

    # 4. Save to Database History
    log_id = None
    if log_to_db:
        try:
            log_id = log_classification(
                filename=filename,
                predicted_style=predicted_style,
                confidence=confidence,
                probabilities=probabilities,
                notes="Inference executed successfully"
            )
        except Exception as e:
            print(f"Warning: Failed to log classification to database: {e}")

    return {
        "predicted_style": predicted_style,
        "confidence": confidence,
        "probabilities": probabilities,
        "raw_scores": [float(p) for p in predictions],
        "is_trained_model": True,
        "log_id": log_id,
        "pil_image": pil_img
    }


def mock_predict_for_setup_preview(
    image_input: Any,
    filename: str = "preview.jpg"
) -> Dict[str, Any]:
    """
    Demo/Fallback predictor used ONLY when the application is running in 'Setup Mode'
    prior to full model training, explicitly identified to the user with a warning banner.
    Does NOT fabricate random numbers silently.
    """
    pil_img = None
    try:
        _, pil_img = load_and_preprocess_image(image_input)
    except Exception:
        pass

    # Clearly labeled setup preview
    demo_probs = {
        "Impressionism": 25.0,
        "Cubism": 25.0,
        "Abstract": 25.0,
        "Surrealism": 25.0
    }

    return {
        "predicted_style": "Impressionism",
        "confidence": 25.0,
        "probabilities": demo_probs,
        "raw_scores": [0.25, 0.25, 0.25, 0.25],
        "is_trained_model": False,
        "setup_notice": "APPLICATION IN SETUP STATE: Model has not yet been trained on your machine.",
        "log_id": None,
        "pil_image": pil_img
    }
