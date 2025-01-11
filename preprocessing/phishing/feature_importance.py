import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the .arff file
data = arff.loadarff('/Users/kweiss/git/cyber/ANTS/data/phishing/phishing_data.arff')
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
X = df.drop('Result_1', axis=1)
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

# Determine feature importance
importance = bst.get_score(importance_type='weight')

# Create a DataFrame to map feature names to their importance scores
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])

# Map feature names to actual feature names
feature_names = X.columns.tolist()
importance_df['Feature'] = importance_df['Feature'].apply(lambda x: feature_names[int(x[1:])])  # Convert f0, f1, ... to actual names

# Save feature importance to a CSV file
importance_df.to_csv('feature_importance_phishing.csv', index=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance from XGBoost')
plt.show()