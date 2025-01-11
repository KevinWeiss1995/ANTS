import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the .arff file
data = arff.loadarff('/Users/kweiss/git/cyber/ANTS/data/phishing/phishing_data.arff')
df = pd.DataFrame(data[0])

# Inspect the DataFrame columns
print("Columns in the DataFrame:", df.columns)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Check for the 'Label' column and strip any whitespace
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
if 'Label' not in df.columns:
    raise KeyError("The 'Label' column is not found in the DataFrame. Available columns: {}".format(df.columns.tolist()))

# Split the data into features and target variable
X = df.drop('Label', axis=1)
y = df['Label']

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
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
plt.xlabel('Feature Importance')
plt.title('Feature Importance from XGBoost')
plt.show()
