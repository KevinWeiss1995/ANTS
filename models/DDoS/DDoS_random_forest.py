import numpy as np
import pandas as pd
import time
import csv
import pickle
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


"""
This module provides functionality to train and evaluate a Random Forest classifier for detecting DDoS attacks 
using network flow features. The script includes preprocessing, feature scaling, class balancing, model training, 
evaluation, and result saving.

Key Functions:
- `train_ddos_model`: Trains a Random Forest classifier on labeled network traffic data, evaluates its performance, 
  and saves the trained model and scaler to disk.
- Main Section: Loads the dataset, preprocesses labels, analyzes data samples, and initiates the training process.

Usage:
1. Update the `DATA_PATH` variable to point to the dataset file.
2. Ensure the `algorithms_features` dictionary contains the desired feature columns for training.
3. Run the script to train the model and save the results.

Features:
- Handles imbalanced datasets using RandomUnderSampler.
- Stratified train-test splitting for balanced evaluation.
- Detailed model performance metrics (accuracy, precision, recall, F1-score).
- Saves the trained model, scaler, and performance metrics to specified locations.

Dependencies:
- NumPy
- Pandas
- scikit-learn
- imbalanced-learn
- pickle
- csv
- os
"""


def get_git_repo_root():
    """Dynamically determine the Git repository root."""
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return None

# Get the repository root
repo_root = get_git_repo_root()



def train_ddos_model(df, y, algorithms_features, result_path=None):
    # If no result_path provided, use the default path
    if result_path is None:
        # result_path = "/Users/kweiss/git/ANTS/results/models"
        result_path = os.join(repo_root, "results", "models")

    # Initialize Random Forest with updated parameters
    rf_classifier = RandomForestClassifier(
        n_estimators=1000,         # Increased from 500
        max_depth=10,              # Reduced from 20 to prevent overfitting
        min_samples_split=10,      # Increased to require more evidence
        min_samples_leaf=8,        # Increased for better generalization
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        class_weight={             # Custom class weights
            0: 3,                  # Weight attack samples more heavily
            1: 1                   # Normal weight for benign
        },
        bootstrap=True,
        oob_score=True
    )

    X = df[algorithms_features['RandomForest']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    # Calculate class distribution
    n_samples_attack = sum(y_train == 0)
    n_samples_benign = sum(y_train == 1)
    
    print(f"\nOriginal class distribution in training set:")
    print(f"Attack samples (0): {n_samples_attack}")
    print(f"Benign samples (1): {n_samples_benign}")

    # More aggressive sampling strategy
    sampling_strategy = {
        0: n_samples_attack,        # Keep all attack samples
        1: n_samples_attack        # Make classes exactly equal
    }

    # Use Random Under Sampling
    sampling_pipeline = Pipeline([
        ('rus', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42))
    ])

    # Resample
    print("\nResampling the training data...")
    X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)
    print(f"Training set shape after resampling: {X_train_resampled.shape}")
    print(f"Class distribution after resampling: {np.bincount(y_train_resampled)}")

    # Train model
    print("\nTraining model...")
    second = time.time()  # Initialize timer here
    rf_classifier.fit(X_train_resampled, y_train_resampled)
    predict = rf_classifier.predict(X_test)

    # Calculate metrics
    f_1 = f1_score(y_test, predict, average='macro')
    pr = precision_score(y_test, predict, average='macro')
    rc = recall_score(y_test, predict, average='macro')
    acc = rf_classifier.score(X_test, y_test)
    training_time = time.time() - second  # Now this will work

    # Print detailed results
    print('\nRandom Forest Model Performance:')
    print(f'Accuracy: {acc:.2f}')
    print(f'Precision: {pr:.2f}')
    print(f'Recall: {rc:.2f}')
    print(f'F1 Score: {f_1:.2f}')
    print(f'Training Time: {training_time:.4f} seconds')
    if hasattr(rf_classifier, 'oob_score_'):
        print(f'Out-of-Bag Score: {rf_classifier.oob_score_:.2f}')

    # Print confusion matrix
    cm = confusion_matrix(y_test, predict)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predict))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': algorithms_features['RandomForest'],
        'importance': rf_classifier.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))

    # Create results directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)

    # Save the model and scaler
    model_filename = os.path.join(result_path, "DDoS_RandomForest.pkl")
    scaler_filename = os.path.join(result_path, "DDoS_RandomForest_scaler.pkl")

    with open(model_filename, "wb") as model_file:
        pickle.dump(rf_classifier, model_file)

    with open(scaler_filename, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"\nModel saved as: {model_filename}")
    print(f"Scaler saved as: {scaler_filename}")

    # Save metrics to CSV
    results_file = os.path.join(result_path, "DDoS_results.csv")
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Training_Time'])
        writer.writerow(['RandomForest', acc, pr, rc, f_1, training_time])

    return rf_classifier, scaler

# Add main section here
if __name__ == "__main__":
    # Define absolute path to data
    # DATA_PATH = "/Users/kweiss/git/ANTS/data/DDoS/all_data.csv"
    DATA_PATH = os.join(repo_root, "data","DDoS","all_data.csv")

    # Define the features for Random Forest
    algorithms_features = {
        'RandomForest': [
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Total Length of Fwd Packets",
            "Fwd Packet Length Std",
            "Flow IAT Std",
            "Flow IAT Min",
            "Fwd IAT Total"
        ]
    }

    # Load the dataset with low_memory=False to avoid DtypeWarning
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print("Loaded dataset successfully")
        print(f"Dataset shape: {df.shape}")
        
        # Print unique values in the label column before conversion
        print("\nUnique values in label column:")
        print(df.iloc[:, -1].unique())
        
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}")
        exit(1)

    # Convert labels to numeric values
    # IP addresses are attacks (0), '0' and '0.0' are benign (1)
    y = (df.iloc[:, -1].isin(['0', '0.0'])).astype(int).values
    
    print("\nLabel conversion:")
    print("IP addresses -> 0 (Attack)")
    print("0 or 0.0 -> 1 (Benign)")
    print(f"\nClass distribution:\n{np.unique(y, return_counts=True)}")

    # Before training, let's analyze some samples
    print("\nAnalyzing sample data:")
    
    # Get attack samples (where y == 0)
    attack_indices = np.where(y == 0)[0][:5]
    print("\nSample ATTACK traffic (first 5):")
    for idx in attack_indices:
        print(f"\nFeatures for attack sample {idx}:")
        for feature_name, value in zip(algorithms_features['RandomForest'], df[algorithms_features['RandomForest']].iloc[idx]):
            print(f"{feature_name}: {value}")

    # Get benign samples (where y == 1)
    benign_indices = np.where(y == 1)[0][:5]
    print("\nSample BENIGN traffic (first 5):")
    for idx in benign_indices:
        print(f"\nFeatures for benign sample {idx}:")
        for feature_name, value in zip(algorithms_features['RandomForest'], df[algorithms_features['RandomForest']].iloc[idx]):
            print(f"{feature_name}: {value}")

    # Continue with model training
    print("\nStarting model training...")
    model, scaler = train_ddos_model(df, y, algorithms_features)
    print("Training complete!")
