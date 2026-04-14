"""
Export the trained Keras digit classifier to TensorFlow Lite for on-device use.
"""

from pathlib import Path

import tensorflow as tf
from tensorflow import keras


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "backend" / "digit_model.h5"
OUTPUT_PATH = ROOT_DIR / "flutter_app" / "assets" / "models" / "digit_model.tflite"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    OUTPUT_PATH.write_bytes(tflite_model)
    print(f"TFLite model exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
