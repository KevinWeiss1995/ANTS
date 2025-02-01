import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import subprocess
import os



def get_git_repo_root():
    """Dynamically determine the Git repository root."""
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return None

base_path = get_git_repo_root()

data_path = os.path.join(base_path, 'data/DDoS/TRAINING_DATA.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Preprocess the data
# Convert categorical labels to numerical
df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})

# Define features and target using only the top 20 most important features
important_features = [
    'Source Port',
    'Bwd Packet Length Mean',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Max',
    'Packet Length Std',
    'Avg Fwd Segment Size',
    'Bwd Packet Length Max',
    'Avg Bwd Segment Size',
    'Bwd Packet Length Std',
    'act_data_pkt_fwd',
    'Subflow Fwd Bytes',
    'Average Packet Size',
    'Packet Length Variance',
    'Total Backward Packets',
    'Max Packet Length',
    'Fwd IAT Std',
    'Bwd Packets/s',
    'Packet Length Mean',
    'Fwd IAT Max',
    'Subflow Fwd Packets'
]

X = df[important_features]  # Use only the important features
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # Handle class imbalance
    n_estimators=100,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model
output_dir = os.path.join(base_path, 'results/models/DDoS')
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
model_path = os.path.join(output_dir, 'DDoS_XGBoost.pkl')
joblib.dump(model, model_path)

print(f'Model saved to {model_path}')
