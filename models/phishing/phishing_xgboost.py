import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Load the .arff file
def load_data(file_path):
    data = loadarff(file_path)
    df = pd.DataFrame(data[0])
    # Convert bytes to strings for categorical columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)
    return df

# Filter and rename features based on importance
def filter_and_rename_features(df, top_features):
    # Filter the dataset to include only the top 10 most important features
    df_filtered = df[top_features + ['Result']]  # Include the target column
    
    # Rename the features to ensure uniqueness
    # Instead of removing suffixes, we keep them to avoid duplicates
    feature_mapping = {feature: feature for feature in top_features}
    df_filtered = df_filtered.rename(columns=feature_mapping)
    
    return df_filtered

# Preprocessing pipeline
def create_preprocessing_pipeline(features):
    # Define numerical features (all features are numerical in this dataset)
    numerical_features = features
    
    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Scale numerical features
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ]
    )
    return preprocessor

# Model training with hyperparameter tuning
def train_model(X_train, y_train, features):
    # Define the model
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Hyperparameter grid for tuning
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Create the full pipeline
    preprocessor = create_preprocessing_pipeline(features)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Randomized search for hyperparameter tuning
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=10,  # Number of parameter settings to sample
        cv=5,       # 5-fold cross-validation
        scoring='accuracy',
        random_state=42,
        n_jobs=-1   # Use all available cores
    )

    # Fit the model
    search.fit(X_train, y_train)
    return search

# Save the model as a pickle file
def save_model(model, file_path):
    joblib.dump(model, file_path)

# Main function
def main():
    # Load data
    file_path = '/Users/kweiss/git/cyber/ANTS/data/phishing/phishing_data.arff'
    df = load_data(file_path)

    # Check for duplicate column names
    duplicates = df.columns[df.columns.duplicated()].unique()
    print("Duplicate columns:", duplicates)

    # Define the top 10 most important features (from your provided data)
    top_features = [
        'URL_of_Anchor',  # Importance: 53.0
        'SSLfinal_State',  # Importance: 46.0
        'Prefix_Suffix',  # Importance: 41.0
        'having_Sub_Domain',  # Importance: 27.0
        'web_traffic',  # Importance: 27.0
        'having_IP_Address',  # Importance: 26.0
        'DNSRecord',  # Importance: 26.0
        'Links_in_tags',  # Importance: 25.0
        'Request_URL'  # Importance: 23.0
    ]

    # Filter and rename features
    df_filtered = filter_and_rename_features(df, top_features)

    # Split data into features and labels
    X = df_filtered.drop('Result', axis=1)
    y = df_filtered['Result']

    # Map target variable from [-1, 1] to [0, 1]
    y = y.map({-1: 0, 1: 1})

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, X.columns.tolist())

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    # Save the model
    save_model(model, 'phishing_model.pkl')
    print("Model saved as 'phishing_model.pkl'")

if __name__ == '__main__':
    main()