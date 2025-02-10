import pandas as pd
import numpy as np
import os
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input



def get_git_repo_root():
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None

base_repo = get_git_repo_root()
data_dir = os.path.join(base_repo, 'data', 'network')

# Load and shuffle data with fixed seed
np.random.seed(42)
train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

# Shuffle training data
shuffle_idx = np.random.permutation(len(train_data))
train_data = train_data.iloc[shuffle_idx].reset_index(drop=True)
train_labels = train_labels.iloc[shuffle_idx].reset_index(drop=True)

# Convert to numpy arrays
X = train_data.values
y = train_labels.values.ravel()
X_test = test_data.values
y_test = test_labels.values.ravel()

# Initialize K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Store fold results
fold_scores = []

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

def create_model(input_shape):
    inputs = Input(shape=(input_shape,), name='input')
    
    # Heavy initial dropout to lower starting accuracy
    x = layers.Dropout(0.7)(inputs)
    
    # Feature extraction path focusing on packet lengths
    x = layers.Dense(48, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01),
                    kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    # Intermediate processing with residual connection
    skip = x
    x = layers.Dense(24, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    # Add attention layer to focus on important features
    attention = layers.Dense(24, activation='tanh')(x)
    attention = layers.Dense(1, activation='softmax')(attention)
    x = layers.Multiply()([x, attention])
    
    x = layers.Dense(48, activation='relu')(x)
    x = layers.Add()([x, skip])  # Residual connection
    
    # Final classification
    x = layers.Dense(16, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print("\nModel Architecture:")
    model.summary()
    
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    print(f"Total non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    
    return model

# After loading data, before model creation
print("\nPreprocessed Data Analysis:")
print(f"Training set shape: {train_data.shape}")
print("\nClass distribution in training set:")
print(train_labels.value_counts(normalize=True))

# Look at feature correlations with target
correlations = []
for column in train_data.columns:
    corr = np.corrcoef(train_data[column], train_labels.values.ravel())[0,1]
    correlations.append((column, abs(corr)))

print("\nTop 10 features by correlation with target:")
for feature, corr in sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"{feature}: {corr:.3f}")

# Basic statistics of preprocessed features
print("\nFeature statistics:")
print(train_data.describe().round(3))

# K-Fold Cross-validation
print("\nPerforming 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f'\nFold {fold + 1}')
    
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]
    
    model = create_model(X.shape[1])
    
    # Calculate step size based on dataset size
    step_size = 8 * (len(X_train_fold) // 32)
    
    # Initialize cyclical learning rate with wider range
    clr = CyclicLR(
        base_lr=0.0005,
        max_lr=0.01,  # Increased max learning rate
        step_size=step_size,
        mode='triangular2'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Add warmup epochs with very high dropout
    warmup_model = create_model(X.shape[1])
    warmup_model.set_weights(model.get_weights())
    warmup_model.layers[1].rate = 0.9  # Increase dropout for warmup
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
    
    # Transfer warmed-up weights
    model.set_weights(warmup_model.get_weights())
    
    print("Main training phase...")
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=50,
        batch_size=32,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stopping, clr],
        verbose=1
    )
    
    scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_scores.append(scores[1])

print(f"\nCross-validation scores: {fold_scores}")
print(f"Mean CV accuracy: {np.mean(fold_scores):.3f} (+/- {np.std(fold_scores) * 2:.3f})")

# Train final model on all training data
final_model = create_model(X.shape[1])
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

final_history = final_model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate final model
predictions = (final_model.predict(X_test) > 0.5).astype(int)
print("\nFinal Model Results:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Create model directory if it doesn't exist
model_dir = os.path.join(base_repo, 'results', 'models', 'network')
os.makedirs(model_dir, exist_ok=True)

# Save the model in Keras format only (more stable)
keras_path = os.path.join(model_dir, 'network_binary_classifier.keras')
final_model.save(keras_path, save_format='keras_v3')

print(f"\nModel saved to: {keras_path}")

# Verify the saved model
loaded_model = keras.models.load_model(keras_path)
test_predictions = (loaded_model.predict(X_test) > 0.5).astype(int)

print("\nVerification of saved model:")
print(classification_report(y_test, test_predictions)) 