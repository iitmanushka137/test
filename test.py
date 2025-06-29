import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def test_model_accuracy_on_sample():
    # Load sample dataset
    df = pd.read_csv("data/sample.csv")
    X = df.drop("species", axis=1)
    y_true_labels = df["species"]

    # Load encoder and scaler
    le = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    y_true = le.transform(y_true_labels)
    X = scaler.transform(X)

    # Reconstruct model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.load_weights("iris_model.weights.h5")

    # Predict and evaluate
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy >= 0.80, f"Accuracy is too low: {accuracy:.2f}"

if __name__ == "__main__":
    test_model_accuracy_on_sample()
    print("✅ Model accuracy test passed.")
