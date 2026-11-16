"""
ARTVISION - Image Preprocessing Module
Handles image validation, resizing (224x224), normalization, format conversion,
and data augmentation pipelines using OpenCV and TensorFlow/Keras.
Includes robust fallback handling for environments during initial setup.
"""

import os
import io
from typing import Tuple, Optional, Union, Dict, Any

# Target dimensions for MobileNetV2 / EfficientNetB0
IMAGE_SIZE: Tuple[int, int] = (224, 224)
TARGET_SHAPE: Tuple[int, int, int] = (224, 224, 3)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit
MIN_DIMENSION = 32  # Minimum acceptable width/height


class ImageValidationError(Exception):
    """Raised when an uploaded or input image fails integrity or format checks."""
    pass


def validate_image_file(file_or_path: Union[str, bytes, io.BytesIO]) -> Tuple[bool, str]:
    """
    Validate that an input is an acceptable, non-corrupted image file.
    
    Returns:
        (is_valid: bool, message: str)
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        # Fallback byte validation when PIL is not yet installed in host environment
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                return False, f"File does not exist: {file_or_path}"
            ext = os.path.splitext(file_or_path)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            if os.path.getsize(file_or_path) == 0:
                return False, "Image file is empty (0 bytes)."
            return True, "File exists and has valid extension."
        elif isinstance(file_or_path, (bytes, bytearray)):
            if len(file_or_path) == 0:
                return False, "Image bytes buffer is empty."
            if len(file_or_path) > MAX_FILE_SIZE_BYTES:
                return False, "Image bytes exceed maximum allowed size."
            return True, "Bytes validation passed."
        elif hasattr(file_or_path, "read"):
            pos = file_or_path.tell() if hasattr(file_or_path, "tell") else 0
            content = file_or_path.read()
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(pos)
            if len(content) == 0:
                return False, "Uploaded image stream is empty."
            return True, "Stream validation passed."
        return True, "Validation passed (fallback mode)."

    try:
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                return False, f"File not found at path: {file_or_path}"
            file_size = os.path.getsize(file_or_path)
            if file_size == 0:
                return False, "File is empty (0 bytes)."
            if file_size > MAX_FILE_SIZE_BYTES:
                return False, f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
            ext = os.path.splitext(file_or_path)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            with Image.open(file_or_path) as img:
                img.verify()
            # Reopen to check dimensions after verify()
            with Image.open(file_or_path) as img:
                w, h = img.size
                if w < MIN_DIMENSION or h < MIN_DIMENSION:
                    return False, f"Image dimensions too small ({w}x{h}). Minimum required: {MIN_DIMENSION}x{MIN_DIMENSION}."
        elif isinstance(file_or_path, bytes):
            if len(file_or_path) == 0:
                return False, "Image bytes buffer is empty."
            if len(file_or_path) > MAX_FILE_SIZE_BYTES:
                return False, "Image bytes exceed maximum allowed size."
            bio = io.BytesIO(file_or_path)
            with Image.open(bio) as img:
                img.verify()
            bio.seek(0)
            with Image.open(bio) as img:
                w, h = img.size
                if w < MIN_DIMENSION or h < MIN_DIMENSION:
                    return False, f"Image dimensions too small ({w}x{h})."
        elif hasattr(file_or_path, "read"):
            # File-like object (e.g. Streamlit UploadedFile)
            pos = file_or_path.tell() if hasattr(file_or_path, "tell") else 0
            content = file_or_path.read()
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(pos)
            if len(content) == 0:
                return False, "Uploaded image stream is empty."
            bio = io.BytesIO(content)
            with Image.open(bio) as img:
                img.verify()
            bio.seek(0)
            with Image.open(bio) as img:
                w, h = img.size
                if w < MIN_DIMENSION or h < MIN_DIMENSION:
                    return False, f"Image dimensions too small ({w}x{h})."
        return True, "Image is valid and uncorrupted."
    except UnidentifiedImageError:
        return False, "The file could not be recognized as a valid image format."
    except Exception as e:
        return False, f"Image validation error: {str(e)}"


def load_and_preprocess_image(
    image_input: Union[str, bytes, io.BytesIO, Any],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    normalization: str = "mobilenet"
):
    """
    Load an image, convert to RGB, resize to target_size (224x224),
    and normalize pixel values.
    
    Args:
        image_input: File path, bytes, BytesIO, or PIL/OpenCV image.
        target_size: (height, width) tuple, defaults to (224, 224).
        normalization: 'mobilenet' (maps to [-1, 1]), 'zero_one' ([0, 1]), or 'standard' (z-score).
        
    Returns:
        numpy.ndarray of shape (1, 224, 224, 3) ready for model prediction,
        and raw RGB PIL image for display.
    """
    # Validate first
    is_valid, msg = validate_image_file(image_input)
    if not is_valid:
        raise ImageValidationError(msg)

    try:
        from PIL import Image
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Required image processing library missing: {e}. Please install Pillow and numpy.")

    # Load image into PIL RGB
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif hasattr(image_input, "read"):
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        pil_img = Image.open(image_input).convert("RGB")
    elif hasattr(image_input, "convert"):
        pil_img = image_input.convert("RGB")
    else:
        # Check if already a numpy array (e.g. from OpenCV)
        try:
            import numpy as np
            if isinstance(image_input, np.ndarray):
                if len(image_input.shape) == 2:  # Grayscale
                    pil_img = Image.fromarray(image_input).convert("RGB")
                elif image_input.shape[2] == 4:  # RGBA
                    pil_img = Image.fromarray(image_input).convert("RGB")
                elif image_input.shape[2] == 3:  # BGR or RGB
                    pil_img = Image.fromarray(image_input[:, :, ::-1]).convert("RGB")
                else:
                    raise ValueError(f"Unexpected image array shape: {image_input.shape}")
            else:
                raise ValueError("Unsupported image input type.")
        except Exception as e:
            raise ImageValidationError(f"Could not convert input to RGB image: {e}")

    # Resize image to target dimensions (using high-quality Lanczos resampling)
    resized_pil = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(resized_pil, dtype=np.float32)

    # Pixel normalization
    if normalization == "mobilenet":
        # MobileNetV2 expects values in [-1, 1]: (x / 127.5) - 1.0
        img_array = (img_array / 127.5) - 1.0
    elif normalization == "zero_one":
        img_array = img_array / 255.0
    elif normalization == "standard":
        mean = np.mean(img_array, axis=(0, 1), keepdims=True)
        std = np.std(img_array, axis=(0, 1), keepdims=True) + 1e-7
        img_array = (img_array - mean) / std

    # Expand batch dimension -> (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)

    return batch_array, pil_img


def load_image_opencv(image_path: str, target_size: Tuple[int, int] = IMAGE_SIZE):
    """
    Alternative loader using OpenCV for image resizing and color space conversion.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        # Fallback to PIL
        return load_and_preprocess_image(image_path, target_size)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Read BGR
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise ImageValidationError(f"OpenCV failed to decode image at {image_path}")

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)

    # Normalize [-1, 1]
    norm_img = (resized_img.astype(np.float32) / 127.5) - 1.0
    batch_array = np.expand_dims(norm_img, axis=0)

    return batch_array, rgb_img


def get_data_augmentation_pipeline():
    """
    Constructs a TensorFlow / Keras Sequential data augmentation pipeline.
    Applies rotation, horizontal flip, zoom, and contrast adjustment.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),  # +/- 15% (~27 degrees)
            layers.RandomZoom(0.1),       # Zoom in/out by up to 10%
            layers.RandomContrast(0.1),   # Contrast variation
        ], name="artwork_augmentation")

        return data_augmentation
    except ImportError:
        return None
