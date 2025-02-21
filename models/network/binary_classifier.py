import pandas as pd
import numpy as np
import os
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, backend as K

def get_git_repo_root():
    """
    Finds the root directory of the git repository.
    
    Returns:
        String path to repository root or None if not in a git repo
    """
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None

base_repo = get_git_repo_root()
data_dir = os.path.join(base_repo, 'data', 'network')

np.random.seed(42)
train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

shuffle_idx = np.random.permutation(len(train_data))
train_data = train_data.iloc[shuffle_idx].reset_index(drop=True)
train_labels = train_labels.iloc[shuffle_idx].reset_index(drop=True)

X = train_data.values
y = train_labels.values.ravel()
X_test = test_data.values
y_test = test_labels.values.ravel()

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []

# Store feature names
feature_names = train_data.columns.tolist()
feature_file_path = os.path.join(base_repo, 'results', 'models', 'network', 'network_features.txt')
os.makedirs(os.path.dirname(feature_file_path), exist_ok=True)

print("\nSaving feature names to:", feature_file_path)
with open(feature_file_path, 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

print("\nFeatures saved:")
for feature in feature_names:
    print(f"- {feature}")

class CyclicLR(keras.callbacks.Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.cycle = 0
        self.step_in_cycle = 0
        
    def on_batch_begin(self, batch, logs=None):
        cycle = np.floor(1 + batch / (2 * self.step_size))
        x = np.abs(batch / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / (2**(cycle - 1))
        
        self.model.optimizer.lr = lr

def focal_loss(gamma=2., alpha=.25):
    """
    Creates a focal loss function for training deep neural networks with imbalanced datasets.
    
    Focal loss reduces the relative loss for well-classified examples and focuses more on
    hard, misclassified examples. Useful when some examples are easy to classify while
    others are hard.
    
    Args:
        gamma: Focus parameter that reduces the loss for well-classified examples.
              Higher values mean more focus on hard examples. Default is 2.
        alpha: Weight parameter for class imbalance. Default is 0.25.
    
    Returns:
        A loss function that can be used in model.compile()
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon()) +
                      (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def create_model(input_shape):
    """
    Creates a neural network for binary classification of network traffic.
    
    Uses a residual architecture with dropout and batch normalization for 
    better feature learning and regularization.
    
    Args:
        input_shape: Number of input features
    
    Returns:
        A compiled Keras model
    """
    inputs = Input(shape=(input_shape,), name='input')
    
    x = layers.Dropout(0.5)(inputs)
    
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.005),
                    kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    skip = x
    x = layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Add()([x, skip])
    
    x = layers.Dense(16, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

print("\nPerforming 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f'\nFold {fold + 1}')
    
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]
    
    model = create_model(X.shape[1])
    print("\nModel Architecture:")
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(gamma=3.0, alpha=0.3),
        metrics=['accuracy', keras.metrics.AUC(),
                keras.metrics.Precision(), 
                keras.metrics.Recall()]
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    warmup_model = create_model(X.shape[1])
    warmup_model.set_weights(model.get_weights())
    warmup_model.layers[1].rate = 0.9
    warmup_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Warmup phase...")
    warmup_model.fit(
        X_train_fold, y_train_fold,
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    model.set_weights(warmup_model.get_weights())
    
    print("\nMain training phase...")
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=100,
        batch_size=64,  
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stopping],
        verbose=1
    )
    
    scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_scores.append(scores[1])

print(f"\nCross-validation scores: {fold_scores}")
print(f"Mean CV accuracy: {np.mean(fold_scores):.3f} (+/- {np.std(fold_scores) * 2:.3f})")


final_model = create_model(X.shape[1])
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

final_history = final_model.fit(
    X, y,
    epochs=100,  # Number of epochs, 100 is high, early stopping will stop training before this
    batch_size=64, 
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

predictions = (final_model.predict(X_test) > 0.5).astype(int)
print("\nFinal Model Results:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
model_dir = os.path.join(base_repo, 'results', 'models', 'network', 'saved_model')
tf.saved_model.save(final_model, model_dir)

print(f"\nModel saved to: {model_dir}")

# Verify saved model - Fix for data type mismatch
loaded_model = tf.saved_model.load(model_dir)
infer = loaded_model.signatures["serving_default"]
# Convert input to float32
X_test_float32 = tf.cast(X_test, tf.float32)
test_predictions = (infer(inputs=X_test_float32)['output_0'] > 0.5).numpy().astype(int)

print("\nVerification of saved model:")
print(classification_report(y_test, test_predictions)) 