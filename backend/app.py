import os
import io
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from utils import predict
from stats import get_model_stats

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Allow cross-origin requests

@app.route('/')
def index():
    """Serve the frontend dashboard."""
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Predict digits drawn on the web canvas."""
    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            logger.warning("Empty request or missing image data")
            return jsonify({
                "error": "No image data provided. Must send base64 string under 'image' key.",
                "status": "failure"
            }), 400

        base64_img = data['image']
        # Strip header if present e.g., 'data:image/png;base64,...'
        if ',' in base64_img:
            base64_img = base64_img.split(',')[1]

        # Decode and load
        try:
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as decode_err:
            logger.error(f"Image decoding failed: {str(decode_err)}")
            return jsonify({
                "error": "Invalid base64 image data",
                "status": "failure"
            }), 400

        # Perform Inference
        logger.info("Executing prediction...")
        result = predict(img)
        result["api_status"] = "success"

        return jsonify(result)

    except Exception as e:
        logger.exception("An unexpected error occurred during prediction")
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "status": "failure"
        }), 500

@app.route('/health', methods=['GET'])
def health_endpoint():
    return jsonify({
        "status": "ok",
        "message": "MLP Digit Classifier API is running!",
        "version": "2.0.0"
    })

@app.route('/api/stats', methods=['GET'])
def stats_endpoint():
    """Returns model performance metrics and confusion matrices."""
    stats = get_model_stats()
    if stats and stats.get("status") == "success":
        return jsonify(stats)
    return jsonify({
        "error": "Could not compute model statistics",
        "status": "failure"
    }), 500

if __name__ == "__main__":
    # Use environment variable for port (Render compatibility)
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
