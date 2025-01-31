import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import time
import csv
import pickle
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
from DDoS_dropout import DropoutRowwise, DropoutColumnwise
from models.DDoS.DDoS_attention import SelfAttentionWithGate, CrossAttentionNoGate


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
    """Neural Network for DDoS detection with attention and custom dropout.
    
    Args:
        input_size: Number of input features.
    """
    def __init__(self, input_size):
        super(DDoSNet, self).__init__()
        
        # Attention configuration
        self.hidden_dim = 64
        self.num_heads = 4
        self.attention_dim = 32
        self.inf = 1e9
        
        self.input_projection = nn.Linear(input_size, self.hidden_dim)
        self.self_attention = SelfAttentionWithGate(
            c_qkv=self.hidden_dim,
            c_hidden=self.attention_dim,
            num_heads=self.num_heads,
            inf=self.inf,
            chunk_size=None
        )
        
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            DropoutRowwise(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            DropoutColumnwise(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        
        # Create attention mask
        batch_size = x.size(0)
        attention_mask = torch.ones(
            (batch_size, self.num_heads, 1, 1),
            device=x.device
        )
        
        x = self.self_attention(
            input_qkv=x,
            mask=attention_mask,
            bias=None
        )
        
        x = x.squeeze(1)
        return self.layers(x)


def train_ddos_model(df, y, algorithms_features, result_path=None):
    # If no result_path provided, use the default path
    if result_path is None:
        result_path = os.path.join(repo_root, "results", "models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = df[algorithms_features['NeuralNetwork']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    # Calculate class distribution
    n_samples_attack = sum(y_train == 0)
    n_samples_benign = sum(y_train == 1)
    
    print(f"\nOriginal class distribution in training set:")
    print(f"Attack samples (0): {n_samples_attack}")
    print(f"Benign samples (1): {n_samples_benign}")

    sampling_strategy = {
        0: n_samples_attack,
        1: n_samples_attack
    }

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    train_dataset = CustomDataset(X_train_resampled, y_train_resampled)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

    # Results 
    print('\nNeural Network Model Performance:')
    print(f'Accuracy: {acc:.2f}')
    print(f'Precision: {pr:.2f}')
    print(f'Recall: {rc:.2f}')
    print(f'F1 Score: {f_1:.2f}')
    print(f'Training Time: {training_time:.4f} seconds')

    # Confusion matrix
    cm = confusion_matrix(y_test, predict)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predict))

    os.makedirs(result_path, exist_ok=True)

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
    DATA_PATH = os.path.join(repo_root, "data", "DDoS", "TRAINING_DATA.csv")

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

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print("Loaded dataset successfully")
        print(f"Dataset shape: {df.shape}")
        
        print("\nUnique values in label column:")
        print(df.iloc[:, -1].unique())
        
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}")
        exit(1)

    y = (df.iloc[:, -1].isin(['0', '0.0'])).astype(int).values
    
    print("\nLabel conversion:")
    print("IP addresses -> 0 (Attack)")
    print("0 or 0.0 -> 1 (Benign)")
    print(f"\nClass distribution:\n{np.unique(y, return_counts=True)}")

  
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


    print("\nStarting model training...")
    model, scaler = train_ddos_model(df, y, algorithms_features)
    print("Training complete!")

    result_path = os.path.join(repo_root, "results", "models")
    model_filename = os.path.join(result_path, "DDoS_NeuralNetwork_v2.pt")
    scaler_filename = os.path.join(result_path, "DDoS_NeuralNetwork_scaler_v2.pkl")

    try:
        results_file = os.path.join(result_path, "DDoS_results_v2.csv")
        metrics = {}
        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row[0] == "NeuralNetwork":
                    metrics = {
                        'accuracy': float(row[1]),
                        'precision': float(row[2]),
                        'recall': float(row[3]),
                        'f1': float(row[4]),
                        'time': float(row[5])
                    }
                    break

        if metrics:
            print("\nLoaded Neural Network Model Performance:")
            print(f'Accuracy: {metrics["accuracy"]:.2f}')
            print(f'Precision: {metrics["precision"]:.2f}')
            print(f'Recall: {metrics["recall"]:.2f}')
            print(f'F1 Score: {metrics["f1"]:.2f}')
            print(f'Training Time: {metrics["time"]:.4f} seconds')
        else:
            print("\nNo metrics found in results file.")

    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"\nError loading metrics: {str(e)}") 