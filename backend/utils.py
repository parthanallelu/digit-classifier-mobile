"""
utils.py
========
Updated prediction utilities for the trained MLP digit classifier.
Includes image preprocessing stages, confidence logic, and detailed logging.
"""

import os
import numpy as np
from PIL import Image
from tensorflow import keras

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_model.h5")

# Global model cache
_model = None

def load_model() -> keras.Model:
    """Load and cache the trained Keras model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'")
        _model = keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(pil_image: Image.Image):
    """
    Process image with stage-by-stage logging.
    1. Grayscale conversion
    2. Resize to 28x28
    3. Normalization [0, 1]
    4. Reshaping for model input (1, 28, 28)
    """
    logs = []
    stages = {}

    # Stage 1: Grayscale
    gray_img = pil_image.convert("L")
    logs.append("Converted image to grayscale.")
    stages["grayscale"] = "Success"

    # Stage 2: Resize
    # Note: We resize to 28x28 as per requirements
    resized_img = gray_img.resize((28, 28), Image.LANCZOS)
    logs.append("Resized image to 28x28.")
    stages["resize"] = "28x28"

    # Stage 3: Normalize
    img_array = np.array(resized_img, dtype="float32") / 255.0
    logs.append("Normalized pixel values to [0, 1].")
    stages["normalization"] = "0.0 - 1.0"

    # Stage 4: Reshape
    # The model expects (batch, width, height) i.e. (1, 28, 28)
    processed_array = np.expand_dims(img_array, axis=0)
    logs.append(f"Reshaped array to {processed_array.shape}.")
    stages["reshape"] = str(processed_array.shape)

    return processed_array, logs, stages

def predict(pil_image: Image.Image) -> dict:
    """
    Run inference with custom confidence logic and return detailed JSON.
    """
    model = load_model()
    
    # Preprocessing
    processed_arr, logs, stages = preprocess_image(pil_image)
    
    # Inference
    logs.append("Running model inference...")
    probs = model.predict(processed_arr, verbose=0)[0]
    probabilities = probs.tolist()
    
    # Get top predictions
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])
    
    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx])
    
    # Confidence Logic
    # 1. IF max_probability < 0.7: "Not a digit"
    # 2. ELSE IF (top1 - top2) < 0.2: "Uncertain"
    # 3. ELSE: return predicted digit
    
    prediction = "Uncertain"
    if top1_prob < 0.7:
        prediction = "Not a digit"
        logs.append(f"Confidence {top1_prob:.2f} is below 0.7 threshold. Result: 'Not a digit'.")
    elif (top1_prob - top2_prob) < 0.2:
        prediction = "Uncertain"
        logs.append(f"Margin between top two values ({top1_prob:.2f} - {top2_prob:.2f} = {top1_prob-top2_prob:.2f}) is below 0.2. Result: 'Uncertain'.")
    else:
        prediction = str(top1_idx)
        logs.append(f"Prediction successful: {prediction} with {top1_prob*100:.1f}% confidence.")

    return {
        "prediction": prediction,
        "confidence": top1_prob,
        "probabilities": probabilities,
        "logs": logs,
        "stages": stages
    }
