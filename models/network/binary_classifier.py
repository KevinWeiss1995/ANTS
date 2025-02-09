import pandas as pd
import numpy as np
import os
import subprocess
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import onnx
import tf2onnx

def get_git_repo_root():
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None

base_repo = get_git_repo_root()
data_dir = os.path.join(base_repo, 'data', 'network')

# Load data
train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

# Convert to numpy arrays
X_train = train_data.values
y_train = train_labels.values
X_test = test_data.values
y_test = test_labels.values

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training Neural Network...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
predictions = (model.predict(X_test) > 0.5).astype(int)

print("\nNeural Network Results:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Create model directory if it doesn't exist
model_dir = os.path.join(base_repo, 'results', 'models', 'network')
os.makedirs(model_dir, exist_ok=True)

# Save the model in Keras format with .keras extension
keras_path = os.path.join(model_dir, 'network_binary_classifier.keras')
model.save(keras_path)

# Convert to ONNX
spec = (tf.TensorSpec((None, X_train.shape[1]), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# Save ONNX model
onnx_path = os.path.join(model_dir, 'network_binary_classifier.onnx')
onnx.save_model(onnx_model, onnx_path)

# Verify ONNX model
import onnxruntime
session = onnxruntime.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
onnx_predictions = session.run(None, {input_name: X_test.astype(np.float32)})[0]
onnx_predictions = (onnx_predictions > 0.5).astype(int)

print("\nONNX Model Verification:")
print(classification_report(y_test, onnx_predictions))

print(f"\nModels saved to: {model_dir}") 