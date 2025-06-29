import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Define the same model architecture used during training
def build_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(4,)),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax')  # Assuming 3 classes in Iris dataset
    ])
    return model

# Step 2: Load the sample test data
data_path = "data/sample.csv"
data = pd.read_csv(data_path)

# If the CSV contains labels, separate them
if 'label' in data.columns:
    X = data.drop(columns=['label']).values
    y = data['label'].values
else:
    X = data.values
    y = None

# Step 3: Load model and weights
model = build_model()
model.load_weights("iris_model.weights.h5")

# Step 4: Make predictions
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

# Step 5: Output
for i, pred in enumerate(predicted_classes):
    if y is not None:
        print(f"Sample {i}: Predicted = {pred}, Actual = {y[i]}")
    else:
        print(f"Sample {i}: Predicted = {pred}")
