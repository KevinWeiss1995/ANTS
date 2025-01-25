import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

data = arff.loadarff('/Users/kweiss/git/cyber/ANTS/data/phishing/phishing_data.arff')

df = pd.DataFrame(data[0])
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
df.dropna(inplace=True)
df = pd.get_dummies(df, drop_first=True)
df.columns = df.columns.str.strip()

if 'Result_1' not in df.columns:
    raise KeyError("The 'Result_1' column is not found in the DataFrame. Available columns: {}".format(df.columns.tolist()))

X = df.drop('Result_1', axis=1)
y = df['Result_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1,
    'seed': 42
}
bst = xgb.train(params, dtrain, num_boost_round=100)


importance = bst.get_score(importance_type='weight')

importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])

feature_names = X.columns.tolist()
importance_df['Feature'] = importance_df['Feature'].apply(lambda x: feature_names[int(x[1:])])  # Convert f0, f1, ... to actual names

importance_df.to_csv('feature_importance_phishing.csv', index=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance from XGBoost')
plt.show()