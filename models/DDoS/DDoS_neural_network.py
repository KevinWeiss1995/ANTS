import numpy as np
import pandas as pd
import time
import csv
import pickle
import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset, DataLoader
import warnings


"""
This module provides functionality to train and evaluate a Neural Network for detecting DDoS attacks 
using network flow features. The script includes preprocessing, feature scaling, class balancing, 
model training, evaluation, and result saving.

Key Components:
- CustomDataset: PyTorch Dataset class for handling the network flow data
- DDoSNet: Neural Network architecture for DDoS detection
- train_ddos_model: Main training function with evaluation and model saving
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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DDoSNet(nn.Module):
    def __init__(self, input_size):
        super(DDoSNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        return self.layers(x)


def train_ddos_model(df, y, algorithms_features, result_path=None):
    # If no result_path provided, use the default path
    if result_path is None:
        result_path = os.path.join(repo_root, "results", "models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = df[algorithms_features['NeuralNetwork']]
    
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

    # Sampling strategy
    sampling_strategy = {
        0: n_samples_attack,
        1: n_samples_attack
    }

    # Use Random Under Sampling
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Create datasets and dataloaders
    train_dataset = CustomDataset(X_train_resampled, y_train_resampled)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = DDoSNet(input_size=len(algorithms_features['NeuralNetwork'])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\nTraining model...")
    second = time.time()
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # Print training and validation loss
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(test_loader):.4f}')

    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    predict = np.array(all_predictions)
    y_test = np.array(all_labels)
    
    f_1 = f1_score(y_test, predict, average='macro')
    pr = precision_score(y_test, predict, average='macro')
    rc = recall_score(y_test, predict, average='macro')
    acc = (predict == y_test).mean()
    training_time = time.time() - second

    # Print results
    print('\nNeural Network Model Performance:')
    print(f'Accuracy: {acc:.2f}')
    print(f'Precision: {pr:.2f}')
    print(f'Recall: {rc:.2f}')
    print(f'F1 Score: {f_1:.2f}')
    print(f'Training Time: {training_time:.4f} seconds')

    # Print confusion matrix
    cm = confusion_matrix(y_test, predict)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predict))

    # Create results directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)

    # Save the model and scaler
    model_filename = os.path.join(result_path, "DDoS_NeuralNetwork_v2.pt")
    scaler_filename = os.path.join(result_path, "DDoS_NeuralNetwork_scaler_v2.pkl")

    torch.save(model.state_dict(), model_filename)
    with open(scaler_filename, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"\nModel saved as: {model_filename}")
    print(f"Scaler saved as: {scaler_filename}")

    # Save metrics to CSV
    results_file = os.path.join(result_path, "DDoS_results_v2.csv")
    with open(results_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['NeuralNetwork', acc, pr, rc, f_1, training_time])

    return model, scaler


if __name__ == "__main__":
    # Define absolute path to data
    DATA_PATH = os.path.join(repo_root, "data", "DDoS", "TRAINING_DATA.csv")

    # Define the features
    algorithms_features = {
        'NeuralNetwork': [
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Total Length of Fwd Packets",
            "Fwd Packet Length Std",
            "Flow IAT Std",
            "Flow IAT Min",
            "Fwd IAT Total"
        ]
    }

    # Load the dataset
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print("Loaded dataset successfully")
        print(f"Dataset shape: {df.shape}")
        
        print("\nUnique values in label column:")
        print(df.iloc[:, -1].unique())
        
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}")
        exit(1)

    # Convert labels
    y = (df.iloc[:, -1].isin(['0', '0.0'])).astype(int).values
    
    print("\nLabel conversion:")
    print("IP addresses -> 0 (Attack)")
    print("0 or 0.0 -> 1 (Benign)")
    print(f"\nClass distribution:\n{np.unique(y, return_counts=True)}")

    # Sample analysis
    print("\nAnalyzing sample data:")
    
    attack_indices = np.where(y == 0)[0][:5]
    print("\nSample ATTACK traffic (first 5):")
    for idx in attack_indices:
        print(f"\nFeatures for attack sample {idx}:")
        for feature_name, value in zip(algorithms_features['NeuralNetwork'], 
                                     df[algorithms_features['NeuralNetwork']].iloc[idx]):
            print(f"{feature_name}: {value}")

    benign_indices = np.where(y == 1)[0][:5]
    print("\nSample BENIGN traffic (first 5):")
    for idx in benign_indices:
        print(f"\nFeatures for benign sample {idx}:")
        for feature_name, value in zip(algorithms_features['NeuralNetwork'], 
                                     df[algorithms_features['NeuralNetwork']].iloc[idx]):
            print(f"{feature_name}: {value}")

    # Train model
    print("\nStarting model training...")
    model, scaler = train_ddos_model(df, y, algorithms_features)
    print("Training complete!")

    # Load the state dict
    model.load_state_dict(torch.load(model_filename, weights_only=True))  # Updated to suppress FutureWarning

    # Load the model and scaler
    model_filename = os.path.join(result_path, "DDoS_NeuralNetwork_v2.pt")
    scaler_filename = os.path.join(result_path, "DDoS_NeuralNetwork_scaler_v2.pkl")

    model.load_state_dict(torch.load(model_filename, weights_only=True))
    with open(scaler_filename, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load metrics from CSV
    results_file = os.path.join(result_path, "DDoS_results_v2.csv")
    with open(results_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row[0] == "NeuralNetwork":
                acc = float(row[1])
                pr = float(row[2])
                rc = float(row[3])
                f_1 = float(row[4])
                training_time = float(row[5])

    # Print loaded metrics
    print("\nLoaded Neural Network Model Performance:")
    print(f'Accuracy: {acc:.2f}')
    print(f'Precision: {pr:.2f}')
    print(f'Recall: {rc:.2f}')
    print(f'F1 Score: {f_1:.2f}')
    print(f'Training Time: {training_time:.4f} seconds')

    # Print confusion matrix
    cm = confusion_matrix(y_test, predict)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predict)) 