import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import pickle
import os

class BaseDataPipeline:
    def __init__(self, qt_path=None):
        """
        Base pipeline with common functionality
        Args:
            qt_path: Path to the saved QuantileTransformer pickle file
        """
        self.qt = None
        if qt_path and os.path.exists(qt_path):
            with open(qt_path, 'rb') as f:
                self.qt = pickle.load(f)
    
    def save_transformer(self, path):
        """Save the fitted QuantileTransformer"""
        if self.qt:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.qt, f)

class BinaryClassificationPipeline(BaseDataPipeline):
    def preprocess_data(self, data):
        """
        Preprocess network data for binary classification
        """
        processed_data = data.copy()
        
        # Remove identifying and metadata features
        columns_to_remove = [
            "Flow ID", 
            "Source IP", 
            "Source Port",
            "Destination IP", 
            "Destination Port",
            "Protocol", 
            "Timestamp",
            "External IP"
        ]
        processed_data = processed_data.drop(columns=columns_to_remove, errors='ignore')
        
        # Drop columns with all zeros
        zero_cols = processed_data.columns[(processed_data == 0).all()]
        processed_data = processed_data.drop(columns=zero_cols)
        
        # Remove highly correlated features
        features_to_remove = [
            "Subflow Bwd Packets",
            "Idle Mean",
            "Flow Packets/s",
            "Flow Duration",
            "Total Backward Packets",
            "min_seg_size_forward",
            "Fwd Packet Length Std",
            "Fwd IAT Std",
            "Flow IAT Std",
            "Flow IAT Max",
            "Subflow Fwd Packets",
            "Fwd IAT Max",
            "Idle Min",
            "Total Fwd Packets",
            "Fwd Header Length",
            "Max Packet Length",
            "Total Length of Bwd Packets",
            "Bwd Packet Length Std",
            "Fwd Packet Length Mean",
            "Bwd Packet Length Max",
            "Total Length of Fwd Packets",
            "Bwd Packet Length Mean",
            "Packet Length Mean",
            "Avg Bwd Segment Size",
            "Average Packet Size"
        ]
        processed_data = processed_data.drop(columns=features_to_remove, errors='ignore')
        
        # Convert Label to binary (if Label column exists)
        if "Label" in processed_data.columns:
            processed_data["traffic type"] = processed_data["Label"].map(
                lambda lbl: 1 if lbl != "BENIGN" else 0
            )
            processed_data = processed_data.drop("Label", axis=1)
        
        # Scale the features (excluding the traffic type column)
        if self.qt:
            features = processed_data.drop("traffic type", axis=1, errors='ignore')
            scaled_features = self.qt.transform(features)
            processed_data = pd.DataFrame(scaled_features, columns=features.columns)
            
            # Add back the traffic type column if it exists
            if "traffic type" in data.columns:
                processed_data["traffic type"] = data["traffic type"]
        
        return processed_data
