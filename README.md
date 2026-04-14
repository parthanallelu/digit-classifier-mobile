# ✦ Handwritten Digit Classifier
### Multilayer Perceptron (MLP) · MNIST Dataset · Flutter & Flask API

---

## 📌 Project Description

This project implements a complete handwritten digit recognition system using a **Multilayer Perceptron (MLP)** neural network trained on the **MNIST** dataset. It features a modern, dark-themed **Flutter Mobile App** that lets users draw a digit with their finger and receive an instant, accurately calculated prediction through a robust **Flask REST API**.

The system is structured as a proper modular project with a clear separation of concerns, making it easy to understand, extend, and present as a college mini project.

---

## 🧠 What is MLP? vs CNN

A **Multilayer Perceptron (MLP)** is a fundamental type of feedforward artificial neural network. It consists of:
- **Input layer** — receives raw pixel data (28×28 = 784 values flattened into a 1D array).
- **Hidden layers** — Dense layers with ReLU activation learn increasingly abstract patterns (128 and 64 neurons).
- **Output layer** — 10 neurons with Softmax activation, one per digit (0–9), offering probability distributions.

**MLP vs. CNN (Convolutional Neural Network):**
While an MLP treats an image as a 1D sequence of pixels and loses raw spatial structures, a CNN uses convolutional filters (2D matrices) to learn hierarchical features like edges, curves, and textures. CNNs generally offer higher accuracy on tasks like MNIST (often ~99.5% vs ~98% for MLPs), but MLPs serve as an excellent pedagogical tool for understanding foundational backpropagation and network mechanics without the complexity of spatial convolutions.

---

## 📚 What is MNIST?

**MNIST** (Modified National Institute of Standards and Technology) is the classic benchmark dataset for digit recognition, consisting of:
- 60,000 training images
- 10,000 test images
- 28×28 grayscale pixels per image
- Labels 0 through 9

It is included in `keras.datasets` and downloaded automatically during model training.

---

## 📁 Project Structure

```
project/
│
├── backend/
│   ├── app.py             ← Flask REST API server (/predict)
│   ├── train_model.py     ← MLP training script (MNIST)
│   ├── predict.py         ← Inference utilities + Preprocessing
│   └── digit_model.h5     ← Saved model (generated after training)
│
├── flutter_app/           ← Flutter Mobile Frontend
│   ├── lib/
│   │   ├── main.dart
│   │   ├── models/prediction_result.dart
│   │   ├── screens/home_screen.dart
│   │   ├── services/api_service.dart
│   │   ├── theme/app_theme.dart
│   │   └── widgets/
│   │       ├── drawing_canvas.dart
│   │       ├── prediction_display.dart
│   │       ├── probability_bars.dart
│   │       └── top_predictions.dart
│   └── pubspec.yaml
│
├── main.py                ← Backend Entry Point (CLI manager)
├── requirements.txt       ← Python dependencies
└── README.md
```

---

## 🔁 Architecture & Data Flow

1. **User Input:** The user draws a digit on the Flutter CustomPaint canvas.
2. **Serialization:** The Flutter app converts the canvas drawing to a Base64-encoded PNG image string.
3. **API Transmission:** An HTTP POST request carries the Base64 string to the Flask `/predict` endpoint.
4. **Processing (Backend):** Flask decodes the image, resizes it to 28x28 using Lanczos resampling, converts it to grayscale, and normalizes it.
5. **Inference:** The processed array is passed to the trained MLP `digit_model.h5`.
6. **Result Handling:** The resulting softmax probabilities are sent back down to Flutter as JSON, parsed seamlessly, and presented using state-based UI updates including an animated water-filling graph.

---

## 🚀 Setup & Execution

### 1. Backend Setup

First, navigate to the `project/` directory and install the necessary Python packages. Optional: Use a virtual environment.

```bash
pip install -r requirements.txt
```

Launch the combined training and server pipeline. If the model hasn't been trained yet, this will download MNIST, build the `digit_model.h5`, and automatically spin up the Flask endpoint.

```bash
python main.py
```
*(Server will start on `http://0.0.0.0:5000`)*

### 2. Frontend Setup (Flutter)

In a new terminal window, navigate into the `flutter_app/` directory and get the packages:

```bash
cd flutter_app
flutter pub get
```

Ensure you have a connected device or an emulator running. Build and run the app:

```bash
flutter run
```

*Note: For an Android emulator, the API defaults to `10.0.2.2:5000`. Adjust `lib/services/api_service.dart` if you are using physical external devices vs `localhost`.*

---

## 🎨 UI & Features

| Feature | Description |
|---|---|
| Responsive UI | Clean, dark theme inspired design working elegantly on both tablet and mobile. |
| Drawing Canvas | Custom finger-drawn white strokes over a black container ensuring maximal model compatibility. |
| API Interfacing | Fast networking layer returning rich JSON predicting digit, raw confidence, and probabilities. |
| Probability Bars | 10 dynamically driven vertical bars utilizing `AnimatedContainer` logic acting as a smoothly growing water-fill visualization. |
| Ranking Board | Shows top 3 candidates and percentage weights neatly ranked below the core prediction output. |
