import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, GridSearchCV
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
df = pd.DataFrame(data[0])  # Convert to DataFrame

# Decode byte strings to regular strings
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check available columns for the target variable
print("Available columns:", df.columns.tolist())  # Print available columns for debugging

# Adjust the target variable name based on the actual column names
target_variable = 'Result_1'  # Change this to 'Result_1' based on the available columns

if target_variable not in df.columns:
    raise KeyError(f"The '{target_variable}' column is not found in the DataFrame. Available columns: {df.columns.tolist()}")

# Split the data into features and target variable
X = df[selected_features]  # Use only the selected important features
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best estimator
bst = grid_search.best_estimator_  # Define bst as the best estimator from grid search

# Save the model as a pickle file
with open('xgboost_model.pkl', 'wb') as model_file:  # Changed path to save model
    pickle.dump(bst, model_file)

# Optionally, save the scaler for later use
with open('scaler.pkl', 'wb') as scaler_file:  # Changed path to save scaler
    pickle.dump(scaler, scaler_file)

# Performance Evaluation
y_pred = bst.predict(X_test)  # Make predictions on the test set

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print("Performance Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)