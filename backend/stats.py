import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import os
import logging

logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_model.h5")

_stats_cache = None

def get_model_stats():
    """
    Computes or returns cached Accuracy, Precision, Recall, F1, and Confusion Matrices.
    Uses the official MNIST test dataset.
    """
    global _stats_cache
    if _stats_cache is not None:
        return _stats_cache

    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return None

        # Load model
        logger.info("Loading model for statistics computation...")
        model = keras.models.load_model(MODEL_PATH)
        
        # Load MNIST test data
        logger.info("Loading MNIST test dataset...")
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Preprocess exactly like training
        x_test_norm = x_test.astype('float32') / 255.0
        
        # Predict
        logger.info("Running bulk inference on 10,000 samples...")
        y_pred_probs = model.predict(x_test_norm, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Confusion Matrix (Counts)
        cm = confusion_matrix(y_test, y_pred)
        
        # Confusion Matrix (%)
        # We divide each row by the sum of that row (total actual for that class)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent = (cm_percent * 100).round(2)
        
        _stats_cache = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_percent": cm_percent.tolist(),
            "status": "success"
        }
        logger.info("Statistics computation complete.")
        return _stats_cache
        
    except Exception as e:
        logger.exception(f"Error during statistics computation: {str(e)}")
        return {
            "status": "failure",
            "error": str(e)
        }
