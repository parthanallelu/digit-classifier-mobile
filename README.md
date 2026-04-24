# 🧠 Neural Digit Classifier Dashboard
### MNIST Digit Recognition · 3-Layer MLP · Real-Time Web Dashboard

---

## 📌 Project Overview
The **Neural Digit Classifier** is a professional, end-to-end Machine Learning ecosystem designed to classify handwritten digits (0–9). Originally a mobile concept, this project has been evolved into a **modular web architecture** featuring a premium dark-mode dashboard, a live cloud-hosted inference API, and comprehensive academic verification through interactive notebooks.

This project utilizes a **Deep Multilayer Perceptron (MLP)** neural network with **Batch Normalization** and **Dropout**, trained on the classic **MNIST** dataset. It achieves exceptional accuracy (~98.2%) while maintaining a lightweight footprint for real-time web inference.

---

## ✨ Key Features
- **Professional Web Dashboard**: A high-fidelity, dark-themed UI built with glassmorphism, fluid "water-fill" animations, and responsive layout.
- **Live Cloud Inference**: Backend API is deployed on **Render**, providing global accessibility for real-time predictions.
- **Advanced Confidence Heuristics**: The model is "self-aware"—it detects blank canvases, complex noise, or ambiguous drawings, returning **"Not a digit"** or **"Uncertain"** instead of guessing.
- **Diagnostic Panel**: Live streaming of backend processing stages (cropping, centering, normalization) directly into the UI.
- **Dynamic Performance Metrics**: Integrated `/api/stats` endpoint that serves real-time confusion matrices and evaluation scores.
- **Academic Verification**: A dedicated Jupyter Notebook documenting the entire pipeline, from data augmentation to precision/recall analysis.

---

## 📁 Project Structure
```text
digit-recognition/
├── backend/
│   ├── app.py             # Flask API Entry Point
│   ├── utils.py           # Preprocessing & Reliability Logic
│   ├── stats.py           # Performance Analytics & Metrics
│   ├── requirements.txt   # Backend dependencies
│   ├── Procfile           # Render deployment configuration
│   └── model/
│       └── digit_model.keras # Trained Keras Model (HDF5 format)
├── frontend/
│   ├── index.html         # Dashboard UI Structure
│   ├── style.css          # Professional Dark-Mode Styles
│   └── script.js          # Interactive Logic & API Integration
├── notebook/
│   └── train_model_final.ipynb  # Interactive Training & Evaluation Report
├── static/
│   └── model.png          # Architecture Visualization
└── README.md
```

---

## 🧠 Neural Network Architecture
The system utilizes a deep feedforward MLP optimized for generalization:
1. **Input Layer**: 784 neurons (28×28 pixels flattened).
2. **Hidden Layer 1**: 512 neurons + **BatchNormalization** + ReLU + **Dropout (0.4)**.
3. **Hidden Layer 2**: 256 neurons + **BatchNormalization** + ReLU + **Dropout (0.3)**.
4. **Hidden Layer 3**: 128 neurons + **BatchNormalization** + ReLU + **Dropout (0.2)**.
5. **Output Layer**: 10 neurons + Softmax (Probability Distribution).

*The inclusion of Batch Normalization significantly speeds up convergence and stabilizes the training process.*

---

## 🚀 Setup & Usage

### 1. Live Deployment
The backend is currently hosted live on Render.
- **API Health**: [https://digit-classifier-backend-0qil.onrender.com/health](https://digit-classifier-backend-0qil.onrender.com/health)
- **Stats Endpoint**: [https://digit-classifier-backend-0qil.onrender.com/api/stats](https://digit-classifier-backend-0qil.onrender.com/api/stats)

### 2. Local Exploration
To run the dashboard locally:
1. Navigate to the `frontend/` directory.
2. Open `index.html` in any modern web browser.
3. Draw a digit and click **Predict Digit** to see the magic.

### 3. Backend & Training
To retrain or modify the model:
1. Navigate to `backend/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Use the Jupyter Notebook in `notebook/train_model_final.ipynb` for the full training suite.

---

## 📊 Performance Metrics
The current model (`digit_model.keras`) achieves the following scores on the MNIST test set:
- **Test Accuracy**: 98.24%
- **Precision**: 98.15% (Macro Average)
- **Recall**: 98.12% (Macro Average)
- **F1-Score**: 98.13% (Macro Average)

---

## 🛠️ Technology Stack
- **AI/ML**: Keras, TensorFlow, Scikit-Learn, NumPy.
- **Backend**: Flask (Python 3.12), PIL (Image Processing).
- **Frontend**: Vanilla JavaScript (ES6+), CSS3 (Glassmorphism), HTML5.
- **Infrastructure**: Render (Cloud Deployment), Git.
- **Design**: Google Fonts (Outfit & JetBrains Mono), CSS Gradients.

---
*Created as an advanced implementation of the MNIST Digit Recognition problem.*
