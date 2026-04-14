"""
train_model.py
==============
Trains a Multilayer Perceptron (MLP) on the MNIST handwritten digit dataset.
Saves the trained model as 'digit_model.h5' for use in the GUI application.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "digit_model.h5")
PLOT_PATH   = os.path.join(os.path.dirname(__file__), "training_history.png")
EPOCHS      = 20
BATCH_SIZE  = 128
VALID_SPLIT = 0.1


# ── Data Loading & Preprocessing ───────────────────────────────────────────────

def load_and_preprocess():
    """Load MNIST, normalize pixels to [0,1], one-hot encode labels."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # One-hot encode labels  (e.g. 3  →  [0,0,0,1,0,0,0,0,0,0])
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test  = keras.utils.to_categorical(y_test,  10)

    print(f"Training samples : {x_train.shape[0]}")
    print(f"Test samples     : {x_test.shape[0]}")
    print(f"Input shape      : {x_train.shape[1:]}")
    return (x_train, y_train), (x_test, y_test)


# ── Model Architecture ─────────────────────────────────────────────────────────

def build_model():
    """
    MLP Architecture
    ────────────────
    Input  : 28×28 image  →  784 features (Flatten)
    Hidden : Dense(128, ReLU)  + Dropout(0.3)
    Hidden : Dense(64,  ReLU)  + Dropout(0.2)
    Output : Dense(10,  Softmax)
    """
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28), name="flatten"),
        layers.Dense(128, activation="relu",    name="hidden_1"),
        layers.Dense(64,  activation="relu",    name="hidden_2"),
        layers.Dense(10,  activation="softmax", name="output"),
    ], name="MLP_DigitClassifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, x_train, y_train):
    """Train model with early stopping and learning rate decay."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALID_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_training_plot(history):
    """Save a publication-quality training history chart."""
    epochs = range(1, len(history.history["loss"]) + 1)

    fig = plt.figure(figsize=(14, 5), facecolor="#0d1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    accent  = "#00d4ff"
    accent2 = "#ff6b6b"
    grid_c  = "#1f2937"
    text_c  = "#e2e8f0"

    for ax, train_key, val_key, title, ylabel in [
        (ax1, "loss",     "val_loss",     "Loss",     "Categorical Cross-Entropy"),
        (ax2, "accuracy", "val_accuracy", "Accuracy", "Accuracy"),
    ]:
        ax.set_facecolor("#161b22")
        ax.plot(epochs, history.history[train_key], color=accent,
                linewidth=2.5, label="Train", zorder=3)
        ax.plot(epochs, history.history[val_key],   color=accent2,
                linewidth=2.5, label="Validation", linestyle="--", zorder=3)
        ax.fill_between(epochs, history.history[train_key], alpha=0.15,
                        color=accent, zorder=2)
        ax.set_title(title, color=text_c, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", color=text_c, fontsize=11)
        ax.set_ylabel(ylabel, color=text_c, fontsize=11)
        ax.tick_params(colors=text_c)
        ax.spines[:].set_color(grid_c)
        ax.grid(color=grid_c, linewidth=0.8, zorder=1)
        ax.legend(facecolor="#1f2937", edgecolor=grid_c, labelcolor=text_c)

    fig.suptitle("MLP Training History — MNIST", color=text_c,
                 fontsize=16, fontweight="bold", y=1.02)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Training plot saved -> {PLOT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  MLP Digit Classifier - Training Script")
    print("=" * 50 + "\n")

    (x_train, y_train), (x_test, y_test) = load_and_preprocess()

    model = build_model()
    model.summary()

    print("\nStarting training ...\n")
    history = train(model, x_train, y_train)

    # Evaluate on held-out test set
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{'-'*40}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"{'-'*40}\n")

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    # Save plot
    save_training_plot(history)

    print("\nTraining complete!\n")


if __name__ == "__main__":
    main()
