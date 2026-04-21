"""
utils.py
========
Enhanced prediction utilities for the trained MLP digit classifier.
Includes robust preprocessing (cropping/centering), confidence logic, and stage logging.
"""

import os
import numpy as np
from PIL import Image, ImageOps
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
    Robust preprocessing mirroring MNIST preparation:
    1. Grayscale & Inversion (if needed)
    2. Bounding box cropping
    3. Centering in a 20x20 box inside 28x28 (maintaining aspect ratio)
    4. Normalization
    """
    logs = []
    stages = {}

    # Stage 1: Convert to Grayscale
    gray_img = pil_image.convert("L")
    
    # Auto-Inversion: MNIST and the model expect white digits on black background.
    # The new light-mode UI sends black digits on white background.
    # We check the average brightness of corners to detect background color.
    img_np = np.array(gray_img)
    corners = [img_np[0,0], img_np[0,-1], img_np[-1,0], img_np[-1,-1]]
    if np.mean(corners) > 128:
        gray_img = ImageOps.invert(gray_img)
        logs.append("Phase 1: Light-mode background detected. Inverted to standard MNIST format.")
    else:
        logs.append("Phase 1: Dark-mode background detected. Grayscale conversion successful.")

    # Stage 2: Bounding Box Cropping
    # Get the bounding box of the non-black area
    # Note: Our canvas is black bg, white fg.
    bbox = gray_img.getbbox()
    if bbox:
        # Crop the image to the digit
        digit_crop = gray_img.crop(bbox)
        logs.append(f"Phase 2: Bounding box found and cropped: {bbox}")
        
        # Determine scaling to fit 20x20 (MNIST standard)
        w, h = digit_crop.size
        ratio = 20.0 / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        digit_resized = digit_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create a 28x28 black canvas
        final_img = Image.new("L", (28, 28), 0)
        # Center the 20x20-scaled digit
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        final_img.paste(digit_resized, (paste_x, paste_y))
        logs.append(f"Phase 3: Rescaled and centered digit in 28x28 frame.")
    else:
        # Empty canvas
        final_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)
        logs.append("Phase 2: No bounding box found (empty canvas). Defaulting to simple resize.")

    # Stage 4: Normalize
    img_array = np.array(final_img, dtype="float32") / 255.0
    logs.append("Phase 4: Normalization (0-1) complete.")

    # Stage 5: Reshape for model (1, 28, 28)
    processed_array = np.expand_dims(img_array, axis=0)
    logs.append(f"Phase 5: Reshaped for Neural Core: {processed_array.shape}")

    # Stage 6: Validation Heuristics (Rule-based)
    active_pixels = np.count_nonzero(img_array)
    total_pixels = img_array.size
    density = active_pixels / total_pixels
    
    is_valid_input = True
    if density < 0.01: # Less than 1%
        is_valid_input = False
        logs.append(f"Input Check: Blank or empty canvas detected (Density: {density:.3f})")
    elif density > 0.75: # Increased from 0.50 to allow very bold drawings
        is_valid_input = False
        logs.append(f"Input Check: Complex drawing or shaded area detected (Density: {density:.3f}). Please draw a single digit.")
    else:
        logs.append(f"Input Check: Structural density verified ({density*100:.1f}%)")

    stages["status"] = "Processed" if is_valid_input else "Rejected"
    stages["density"] = float(density)
    
    return processed_array, logs, stages, is_valid_input

def predict(pil_image: Image.Image) -> dict:
    """
    Run inference with custom confidence logic and return detailed JSON.
    """
    model = load_model()
    
    # Preprocessing & Validation
    processed_arr, logs, stages, is_valid_input = preprocess_image(pil_image)
    
    if not is_valid_input:
        return {
            "prediction": "Not a digit",
            "confidence": 0.0,
            "probabilities": [0.0] * 10,
            "status": "invalid",
            "logs": logs,
            "stages": stages
        }

    # Inference
    probs = model.predict(processed_arr, verbose=0)[0]
    probabilities = [float(p) for p in probs]
    
    # Get top predictions
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])
    
    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx])
    
    # Reliability Heuristics
    prediction = str(top1_idx)
    status = "valid"

    if top1_prob < 0.6: # Lowered from 0.7 to handle bold styles
        prediction = "Not a digit"
        status = "invalid"
        logs.append(f"Reliability Check: Max confidence {top1_prob:.2f} < 0.6. Input style is too diverse for single-digit classification.")
    elif (top1_prob - top2_prob) < 0.15: # Slightly loosened margin
        prediction = "Uncertain"
        status = "uncertain"
        logs.append(f"Reliability Check: Ambiguous margin ({top1_prob:.2f} vs {top2_prob:.2f}).")
    else:
        logs.append(f"Success: Match found for '{prediction}' ({top1_prob*100:.1f}%)")

    return {
        "prediction": prediction,
        "confidence": top1_prob,
        "probabilities": probabilities,
        "status": status,
        "logs": logs,
        "stages": stages
    }
