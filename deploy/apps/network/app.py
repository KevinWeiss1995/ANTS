from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as keras
import os
from pipeline.data_pipeline import BinaryClassificationPipeline

app = Flask(__name__)

# Load model and pipeline
def get_model_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'results', 'models', 'network', 'network_binary_classifier.keras')

def get_transformer_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'results', 'transformers', 'quantile_transformer.pkl')

# Load model and pipeline at startup
model = keras.models.load_model(get_model_path())
pipeline = BinaryClassificationPipeline(qt_path=get_transformer_path())

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        json_data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame(json_data)
        
        # Preprocess the data
        processed_data = pipeline.preprocess_data(df)
        
        # Make prediction
        predictions = model.predict(processed_data)
        predictions = (predictions > 0.5).astype(int)
        
        # Return predictions
        return jsonify({
            'predictions': predictions.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=False)
