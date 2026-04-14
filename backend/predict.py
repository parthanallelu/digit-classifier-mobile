"""
predict.py
==========
Prediction utilities for the trained MLP digit classifier.
Provides a clean interface for image preprocessing and inference.
"""

import os

import numpy as np
from PIL import Image
from tensorflow import keras


MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.h5")
MODEL_SIZE = 28
DIGIT_TARGET_SIZE = 20
INK_THRESHOLD = 20

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
    Convert a drawn canvas image into an MNIST-like 28x28 input.

    Pipeline:
        1. Convert to grayscale
        2. Find the bounding box of the drawn ink
        3. Crop around the digit with a small padding
        4. Resize the digit to fit within a 20x20 box while keeping aspect ratio
        5. Center it on a 28x28 black canvas
        6. Normalize pixel values to [0, 1]

    Args:
        pil_image: Raw PIL image from the drawing canvas.

    Returns:
        numpy array of shape (1, 28, 28), dtype float32.
    """
    image = pil_image.convert("L")
    arr = np.array(image, dtype=np.uint8)

    ys, xs = np.where(arr > INK_THRESHOLD)
    if len(xs) == 0 or len(ys) == 0:
      return np.zeros((1, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)

    padding = 12
    left = max(int(xs.min()) - padding, 0)
    right = min(int(xs.max()) + padding, image.width - 1)
    top = max(int(ys.min()) - padding, 0)
    bottom = min(int(ys.max()) + padding, image.height - 1)

    cropped = image.crop((left, top, right + 1, bottom + 1))

    width, height = cropped.size
    if width >= height:
        target_width = DIGIT_TARGET_SIZE
        target_height = max(1, round(height * DIGIT_TARGET_SIZE / width))
    else:
        target_height = DIGIT_TARGET_SIZE
        target_width = max(1, round(width * DIGIT_TARGET_SIZE / height))

    resized = cropped.resize((target_width, target_height), Image.LANCZOS)

    canvas = Image.new("L", (MODEL_SIZE, MODEL_SIZE), color=0)
    offset_x = (MODEL_SIZE - target_width) // 2
    offset_y = (MODEL_SIZE - target_height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    normalized = np.array(canvas, dtype="float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def predict(pil_image: Image.Image) -> dict:
    """
    Run inference on a PIL image and return rich prediction results.

    Args:
        pil_image: PIL image from the drawing canvas.

    Returns:
        dict with keys:
            - digit        (int)   : most likely digit
            - confidence   (float) : probability of the top digit [0-1]
            - probabilities(list)  : 10 probabilities, indices 0-9
            - top3         (list)  : [(digit, prob), ...] top 3 predictions
    """
    model = load_model()
    arr = preprocess_image(pil_image)

    probs = model.predict(arr, verbose=0)[0]

    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(int(i), float(probs[i])) for i in top3_idx]

    return {
        "digit": digit,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "top3": top3,
    }
