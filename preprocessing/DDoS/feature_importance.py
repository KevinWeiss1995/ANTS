import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_feature_importance(data_path):
    """Analyze and visualize feature importance for DDoS detection."""
    
    # Read the dataset
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Convert Label column to binary (DDoS = 0, BENIGN = 1)
    le = LabelEncoder()
    y = le.fit_transform(df['Label'])
    
    # Remove non-numeric columns
    non_numeric_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'External IP', 'Label']
    features = [col for col in df.columns if col not in non_numeric_cols]
    
    print(f"\nAnalyzing {len(features)} numeric features:")
    for f in features:
        print(f"  - {f}")
    
    # Prepare X
    X = df[features]
    
    # Handle any infinite or NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest for feature importance
    print("\nTraining Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Create DataFrame for importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Print top 20 most important features
    print("\nTop 20 Most Important Features:")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Visualize feature importance (top 20)
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features for DDoS Detection')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(data_path), 'feature_importance.png')
    plt.savefig(plot_path)
    print(f"\nFeature importance plot saved to: {plot_path}")
    
    # Analyze distributions for top 10 features
    print("\nFeature Distributions for Top 10 Features (Normal vs DDoS Traffic):")
    for feature in feature_importance['Feature'].head(10):
        normal_values = df[df['Label'] == 'BENIGN'][feature]
        attack_values = df[df['Label'] == 'DDoS'][feature]
        
        print(f"\n{feature}:")
        print("Normal Traffic:")
        print(f"  Mean: {normal_values.mean():.2f}")
        print(f"  Std: {normal_values.std():.2f}")
        print(f"  25th percentile: {normal_values.quantile(0.25):.2f}")
        print(f"  75th percentile: {normal_values.quantile(0.75):.2f}")
        print(f"  Range: [{normal_values.min():.2f}, {normal_values.max():.2f}]")
        
        print("DDoS Traffic:")
        print(f"  Mean: {attack_values.mean():.2f}")
        print(f"  Std: {attack_values.std():.2f}")
        print(f"  25th percentile: {attack_values.quantile(0.25):.2f}")
        print(f"  75th percentile: {attack_values.quantile(0.75):.2f}")
        print(f"  Range: [{attack_values.min():.2f}, {attack_values.max():.2f}]")

if __name__ == "__main__":
    DATA_PATH = "/Users/kweiss/git/cyber/ANTS/data/DDoS/TRAINING_DATA.csv"
    analyze_feature_importance(DATA_PATH)
