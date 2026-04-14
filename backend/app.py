from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
from backend.predict import predict

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Flutter

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Predict digits drawn in the mobile app."""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided. Must send base64 string under 'image' key."}), 400

        base64_img = data['image']
        # Strip header if present e.g., 'data:image/png;base64,...'
        if ',' in base64_img:
            base64_img = base64_img.split(',')[1]

        # Decode and load
        img_bytes = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_bytes))

        # Perform Inference
        result = predict(img)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_endpoint():
    return jsonify({"status": "ok", "message": "MLP Digit Classifier API is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
