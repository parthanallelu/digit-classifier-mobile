import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import os
import logging

logger = logging.getLogger(__name__)

# Paths
def get_model_path():
    """Return the path to the best available model file."""
    keras_path = os.path.join(os.path.dirname(__file__), "model", "digit_model.keras")
    h5_path = os.path.join(os.path.dirname(__file__), "model", "digit_model.h5")
    return keras_path if os.path.exists(keras_path) else h5_path

_stats_cache = None
_model_last_modified = 0
_current_model_path = None

def get_model_stats():
    """
    Computes or returns cached Accuracy, Precision, Recall, F1, and Confusion Matrices.
    Uses the official MNIST test dataset. Auto-refreshes if the model file changes.
    """
    global _stats_cache, _model_last_modified, _current_model_path
    
    try:
        model_path = get_model_path()
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None

        # Check modified time or file path change for auto-refresh
        mtime = os.path.getmtime(model_path)
        if (_stats_cache is not None and 
            mtime <= _model_last_modified and 
            model_path == _current_model_path):
            return _stats_cache
        
        _model_last_modified = mtime
        _current_model_path = model_path

        # Load model
        logger.info(f"Loading model for statistics computation from {model_path}...")
        model = keras.models.load_model(model_path)
        
        # Load MNIST test data
        logger.info("Loading MNIST test dataset...")
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Preprocess exactly like training
        x_test_norm = x_test.astype('float32') / 255.0
        
        # Predict
        logger.info("Running bulk inference on 10,000 samples...")
        y_pred_probs = model.predict(x_test_norm, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate Metrics (matching notebook macro-averaging)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Confusion Matrix (Counts)
        cm = confusion_matrix(y_test, y_pred)
        
        # Confusion Matrix (%)
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

