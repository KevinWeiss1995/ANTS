import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle  # Import the pickle module
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Import evaluation metrics

# Load the feature importance data
importance_df = pd.read_csv('/Users/kweiss/git/cyber/ANTS/data/phishing/feature_importance_phishing.csv')  # Updated path

# Set a threshold for selecting important features (e.g., top 10 features)
threshold = importance_df['Importance'].quantile(0.75)  # Select features above the 75th percentile
selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

# Load the original .arff file
data = arff.loadarff('/Users/kweiss/git/cyber/ANTS/data/phishing/phishing_data.arff')  # No change in path
df = pd.DataFrame(data[0])

# Decode byte strings to regular strings
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check available columns for the target variable
if 'Result_1' not in df.columns:
    raise KeyError("The 'Result_1' column is not found in the DataFrame. Available columns: {}".format(df.columns.tolist()))

# Split the data into features and target variable
X = df[selected_features]  # Use only the selected important features
y = df['Result_1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1,
    'seed': 42
}
bst = xgb.train(params, dtrain, num_boost_round=100)

# Save the model as a pickle file
with open('xgboost_model.pkl', 'wb') as model_file:  # Changed path to save model
    pickle.dump(bst, model_file)

# Optionally, save the scaler for later use
with open('scaler.pkl', 'wb') as scaler_file:  # Changed path to save scaler
    pickle.dump(scaler, scaler_file)

# Performance Evaluation
dtest = xgb.DMatrix(X_test)  # Convert test data to DMatrix
y_pred = bst.predict(dtest)  # Make predictions

# Convert probabilities to binary outcomes (if binary classification)
# If multiclass, use argmax to get the predicted class
if len(set(y)) == 2:  # Check if binary classification
    y_pred_binary = [1 if pred > 0.5 else -1 for pred in y_pred]
else:
    y_pred_binary = [int(round(pred)) for pred in y_pred]  # For multiclass, round to nearest integer

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, average='weighted')  # Use 'weighted' for multiclass
recall = recall_score(y_test, y_pred_binary, average='weighted')  # Use 'weighted' for multiclass
f1 = f1_score(y_test, y_pred_binary, average='weighted')  # Use 'weighted' for multiclass
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Print evaluation results
print("Performance Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)