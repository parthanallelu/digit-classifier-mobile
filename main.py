"""
main.py
=======
Entry point for the Handwritten Digit Classifier backend.

Usage:
    python main.py             → Train model (if needed) & Launch API Server
    python main.py --serve     → Run the Flask API Server only
    python main.py --train     → Train the model and exit
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train():
    """Train the MLP model on MNIST."""
    from backend.train_model import main as run_training
    run_training()


def serve():
    """Launch the Flask REST API."""
    from backend.app import app
    print("\nStarting Flask Backend API Server on http://0.0.0.0:5000/ ...\n")
    app.run(host="0.0.0.0", port=5000, debug=False)


def print_help():
    print("""
╔══════════════════════════════════════════════════════╗
║    Handwritten Digit Classifier  ·  Backend API     ║
╚══════════════════════════════════════════════════════╝

Usage:
  python main.py                 Train model (if needed) & Launch API
  python main.py --serve         Launch Flask API Server only
  python main.py --train         Train model only (no API)
  python main.py --help          Show this message
""")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_help()
        sys.exit(0)

    if "--train" in args:
        train()
        sys.exit(0)

    if "--serve" in args:
        serve()
        sys.exit(0)

    # Default action: check model exists, if not train, then serve
    model_path = os.path.join(os.path.dirname(__file__), "backend", "digit_model.h5")
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'. Training automatically...")
        train()
    
    serve()

