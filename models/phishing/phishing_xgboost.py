import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 

importance_df = pd.read_csv('/Users/kweiss/git/cyber/ANTS/data/phishing/feature_importance_phishing.csv') 

threshold = importance_df['Importance'].quantile(0.75)  
selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

data = arff.loadarff('/Users/kweiss/git/cyber/ANTS/data/phishing/phishing_data.arff') 
df = pd.DataFrame(data[0]) 

df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
df.dropna(inplace=True)
df = pd.get_dummies(df, drop_first=True)
df.columns = df.columns.str.strip()

print("Available columns:", df.columns.tolist())  

target_variable = 'Result_1'  
if target_variable not in df.columns:
    raise KeyError(f"The '{target_variable}' column is not found in the DataFrame. Available columns: {df.columns.tolist()}")

X = df[selected_features] 
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

grid_search.fit(X_train, y_train)

bst = grid_search.best_estimator_  

with open('xgboost_model.pkl', 'wb') as model_file: 
    pickle.dump(bst, model_file)

with open('scaler.pkl', 'wb') as scaler_file:  
    pickle.dump(scaler, scaler_file)

y_pred = bst.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  
recall = recall_score(y_test, y_pred, average='weighted')  
conf_matrix = confusion_matrix(y_test, y_pred)

print("Performance Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)