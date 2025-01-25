import json
import requests
import pickle
import os
import numpy as np
import datetime

# Get the path to the saved model
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print(f"### BASE PATH ####:{base_path}")
model_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest.pkl")
scaler_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest_scaler.pkl")

api_url = "http://127.0.0.1:5001/predict"

# Test vectors
test_data = {
    "benign_samples": [
        # Normal web browsing pattern
        [0.2, 2500.0, 12.0, 0.1, 0.5, 4.0, 4.0],
        # Video streaming pattern
        [0.3, 3500.0, 15.0, 0.2, 1.0, 5.0, 5.0],
        # File download pattern
        [0.1, 4000.0, 20.0, 0.3, 0.8, 3.0, 3.0],
        # Regular web API calls
        [0.0, 2000.0, 10.0, 0.0, 0.2, 4.0, 4.0]
    ],
    "attack_samples": [
        # DDoS patterns
        [2500.0, 10000.0, 26.0, 10.0, 400000.0, 2.0, 1000.0],
        [2200.0, 15000.0, 26.0, 13.0, 290000.0, 6.0, 1500.0],
        [2500.0, 20000.0, 27.0, 11.0, 350000.0, 3.0, 1000.0],
        [1500, 10700.0, 20.0, 9.0, 500000.0, 7.0, 1000.0]
    ]
}

def test_prediction(features, expected_type, results_dict):
    """Test prediction and update results dictionary"""
    data = {"features": features}
    response = requests.post(api_url, json=data)
    
    # Create test result dictionary
    test_result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "test_type": expected_type,
        "features": features,
        "expected": expected_type.upper(),
        "api_response": None,
        "model_response": None
    }
    
    print("\n" + "="*50)
    print(f"Testing {expected_type.upper()} pattern...")
    print("-"*50)
    
    # Track predictions
    api_prediction = None
    model_prediction = None
    
    # Test API
    if response.status_code == 200:
        api_result = response.json()
        raw_prediction = api_result.get('prediction', 'Unknown')
        api_prediction = "BENIGN" if str(raw_prediction) == "1" else "ATTACK"
        probability = api_result.get('probability')
        
        test_result["api_response"] = {
            "status_code": response.status_code,
            "prediction": api_prediction,
            "probability": probability if probability is not None else "N/A"
        }
        
        print(f"API Prediction: {api_prediction}")
        if probability is not None:
            print(f"Confidence: {float(probability):.2%}")
        else:
            print("Confidence: N/A")
    else:
        print(f"❌ API Error {response.status_code}: {response.text}")
        test_result["api_response"] = {
            "status_code": response.status_code,
            "error": response.text
        }
    
    # Test model directly
    try:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        X_test = np.array(features).reshape(1, -1)
        X_test_scaled = scaler.transform(X_test)
        
        prediction = clf.predict(X_test_scaled)
        model_prediction = "BENIGN" if prediction[0] == 1 else "ATTACK"
        
        if hasattr(clf, 'predict_proba'):
            probas = clf.predict_proba(X_test_scaled)
            attack_prob = probas[0][0]  # Probability of ATTACK
            benign_prob = probas[0][1]  # Probability of BENIGN
            print(f"Probability of ATTACK: {attack_prob:.2%}")
            print(f"Probability of BENIGN: {benign_prob:.2%}")
            
            test_result["model_response"] = {
                "prediction": model_prediction,
                "probability_attack": float(attack_prob),
                "probability_benign": float(benign_prob)
            }
        
        print(f"\nDirect Model Prediction: {model_prediction}")
            
    except Exception as e:
        print(f"❌ Model Error: {str(e)}")
        test_result["model_response"] = {
            "error": str(e)
        }
    
    # Update results
    expected_prediction = "BENIGN" if expected_type.lower() == "benign" else "ATTACK"
    if api_prediction:
        results_dict['total_api_predictions'] += 1
        if api_prediction == expected_prediction:
            results_dict['correct_api_predictions'] += 1
    
    if model_prediction:
        results_dict['total_model_predictions'] += 1
        if model_prediction == expected_prediction:
            results_dict['correct_model_predictions'] += 1
    
    return test_result

def save_results(all_results):
    """Save test results to a text file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_path, "results", "test_results")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename with timestamp
    filename = os.path.join(results_dir, f"multiple_test_results_{timestamp}.txt")
    
    # Save results to file
    with open(filename, 'w') as f:
        for result in all_results:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Test Type: {result['test_type'].upper()}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
            f.write(f"Features: {result['features']}\n")
            f.write(f"Expected: {result['expected']}\n")
            
            # Write API response
            f.write("\nAPI Response:\n")
            if result['api_response'].get('error'):
                f.write(f"Error: {result['api_response']['error']}\n")
            else:
                f.write(f"Prediction: {result['api_response']['prediction']}\n")
                f.write(f"Probability: {result['api_response']['probability']}\n")
            
            # Write Model response
            f.write("\nDirect Model Response:\n")
            if result['model_response'].get('error'):
                f.write(f"Error: {result['model_response']['error']}\n")
            else:
                f.write(f"Prediction: {result['model_response']['prediction']}\n")
                f.write(f"Probability of ATTACK: {result['model_response']['probability_attack']:.2%}\n")
                f.write(f"Probability of BENIGN: {result['model_response']['probability_benign']:.2%}\n")
            
            f.write("\n")
    
    print(f"\nTest results saved to: {filename}")

def print_summary(results):
    """Print summary of prediction results"""
    print("\n" + "="*50)
    print("TESTING SUMMARY")
    print("="*50)
    
    # API Results
    api_accuracy = (results['correct_api_predictions'] / results['total_api_predictions'] * 100 
                   if results['total_api_predictions'] > 0 else 0)
    print(f"\nAPI Predictions:")
    print(f"Correct: {results['correct_api_predictions']}")
    print(f"Total: {results['total_api_predictions']}")
    print(f"Accuracy: {api_accuracy:.1f}%")
    
    # Model Results
    model_accuracy = (results['correct_model_predictions'] / results['total_model_predictions'] * 100 
                     if results['total_model_predictions'] > 0 else 0)
    print(f"\nDirect Model Predictions:")
    print(f"Correct: {results['correct_model_predictions']}")
    print(f"Total: {results['total_model_predictions']}")
    print(f"Accuracy: {model_accuracy:.1f}%")

def main():
    # Initialize results tracking
    results = {
        'total_api_predictions': 0,
        'correct_api_predictions': 0,
        'total_model_predictions': 0,
        'correct_model_predictions': 0
    }
    
    # Store all test results
    all_test_results = []
    
    print("\nTESTING BENIGN PATTERNS")
    print("="*50)
    for i, sample in enumerate(test_data["benign_samples"], 1):
        print(f"\nBenign Test Case {i}")
        test_result = test_prediction(sample, "benign", results)
        all_test_results.append(test_result)
    
    print("\nTESTING ATTACK PATTERNS")
    print("="*50)
    for i, sample in enumerate(test_data["attack_samples"], 1):
        print(f"\nAttack Test Case {i}")
        test_result = test_prediction(sample, "attack", results)
        all_test_results.append(test_result)
    
    print_summary(results)
    
    # Save all results to file
    save_results(all_test_results)

if __name__ == "__main__":
    main()