"""
predict.py
==========
Prediction utilities for the trained MLP digit classifier.
Provides a clean interface for image preprocessing and inference.
"""

import os
import numpy as np
from tensorflow import keras
from PIL import Image


MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.h5")

# Module-level model cache (loaded once, reused on every call)
_model = None


def load_model() -> keras.Model:
    """Load (and cache) the trained Keras model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'.\n"
                "Please run  model/train_model.py  first."
            )
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image drawn on the canvas into a model-ready numpy array.

    Pipeline:
        1. Convert to grayscale
        2. Resize to 28×28 (MNIST format)  — LANCZOS for best quality
        3. Convert to float32 array
        4. Normalize pixel values to [0, 1]
        5. Add batch dimension  →  shape (1, 28, 28)

    Args:
        pil_image: Raw PIL image from the drawing canvas.

    Returns:
        numpy array of shape (1, 28, 28), dtype float32.
    """
    img = pil_image.convert("L")                       # grayscale
    img = img.resize((28, 28), Image.LANCZOS)           # resize
    arr = np.array(img, dtype="float32") / 255.0        # normalize
    arr = np.expand_dims(arr, axis=0)                   # (1, 28, 28)
    return arr


def predict(pil_image: Image.Image) -> dict:
    """
    Run inference on a PIL image and return rich prediction results.

    Args:
        pil_image: PIL image from the drawing canvas.

    Returns:
        dict with keys:
            - digit        (int)   : most likely digit
            - confidence   (float) : probability of the top digit [0–1]
            - probabilities(list)  : 10 probabilities, indices 0–9
            - top3         (list)  : [(digit, prob), …] top 3 predictions
    """
    model = load_model()
    arr   = preprocess_image(pil_image)

    probs = model.predict(arr, verbose=0)[0]   # shape (10,)

    digit      = int(np.argmax(probs))
    confidence = float(probs[digit])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3     = [(int(i), float(probs[i])) for i in top3_idx]

    return {
        "digit"        : digit,
        "confidence"   : confidence,
        "probabilities": probs.tolist(),
        "top3"         : top3,
    }
