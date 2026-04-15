# 🧠 Neural Digit Classifier Dashboard
### MNIST Digit Recognition · Multilayer Perceptron (MLP) · Live Web Dashboard

---

## 📌 Project Overview
The **Neural Digit Classifier** is a professional, end-to-end Machine Learning ecosystem designed to classify handwritten digits (0–9). Originally a mobile concept, this project has been evolved into a **modular web architecture** featuring a beautiful dark-mode dashboard, a live cloud-hosted inference API, and comprehensive academic verification through interactive notebooks.

This project uses a **Multilayer Perceptron (MLP)** neural network trained on the classic **MNIST** dataset, achieving high accuracy (~98%) while maintaining a lightweight footprint for real-time web inference.

---

## ✨ Key Features
- **Professional Web Dashboard**: A high-fidelity, dark-themed UI built with glassmorphism and fluid "water-fill" animations.
- **Live Cloud Inference**: Backend API is deployed on **Render**, providing global accessibility for real-time predictions.
- **Advanced Confidence Logic**: The model is "self-aware"—it returns **"Not a digit"** for noise or **"Uncertain"** if two digits are too similar, rather than guessing incorrectly.
- **Diagnostic Panel**: Live streaming of backend processing stages and logs directly into the frontend UI.
- **Academic Verification**: A dedicated Jupyter Notebook documenting every training step, including dataset visualization and precision/recall/F1 metrics.

---

## 📁 Project Structure
```text
digit-recognition/
├── backend/
│   ├── app.py             # Flask API Entry Point
│   ├── utils.py           # Preprocessing & Confidence Logic
│   ├── requirements.txt   # Backend dependencies
│   ├── Procfile           # Render deployment configuration
│   └── model/
│       └── digit_model.h5 # Trained Keras Model
├── frontend/
│   ├── index.html         # Dashboard UI Structure
│   ├── style.css          # Professional Dark-Mode Styles
│   └── script.js          # Interactive Logic & API Integration
├── notebook/
│   └── train_model.ipynb  # Interactive Training & Evaluation Report
├── static/
│   └── model.png          # Model Architecture & History Visualization
└── README.md
```

---

## 🧠 Neural Network Architecture (MLP)
The system utilizes a feedforward Multilayer Perceptron:
1. **Input Layer**: 784 neurons (28×28 pixels flattened).
2. **Hidden Layer 1**: 128 neurons (ReLU) with Dropout (0.2).
3. **Hidden Layer 2**: 64 neurons (ReLU) with Dropout (0.2).
4. **Output Layer**: 10 neurons (Softmax) providing probability distributions for digits 0–9.

---

## 🚀 Setup & Usage

### 1. Live Deployment
The backend is currently hosted live on Render. You can access the health check here:
[https://digit-classifier-backend-0qil.onrender.com/health](https://digit-classifier-backend-0qil.onrender.com/health)

### 2. Local Exploration
To run the dashboard locally:
1. Navigate to the `frontend/` directory.
2. Open `index.html` in any modern web browser.
3. Draw a digit and click **Predict Digit**.

### 3. Backend Training (Optional)
If you wish to retrain or modify the model:
1. Navigate to `backend/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Use the Jupyter Notebook in `notebook/train_model.ipynb` for an interactive training experience.

---

## 📊 Performance Metrics
The current model (`digit_model.h5`) achieves the following scores on the MNIST test set:
- **Test Accuracy**: 98.24%
- **Precision**: 98.15% (Macro Average)
- **Recall**: 98.12% (Macro Average)
- **F1-Score**: 98.13% (Macro Average)

---

## 🛠️ Technology Stack
- **Languages**: Python, JavaScript, HTML5, CSS3.
- **Frameworks**: Keras, TensorFlow, Flask, Scikit-Learn.
- **Tools**: Jupyter Notebooks, Git, Render (Deployment).
- **Design**: Vanilla CSS (Glassmorphism), Google Fonts (Outfit & JetBrains Mono).

---
*Created as an advanced implementation of the MNIST Digit Recognition problem.*
