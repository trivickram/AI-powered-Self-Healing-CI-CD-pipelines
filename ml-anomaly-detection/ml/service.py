from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
import pandas as pd
from datetime import datetime
from model import CICDAnomalyDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model instance
detector = None
model_loaded = False

def load_model():
    """
    Load the pre-trained model on startup
    """
    global detector, model_loaded
    
    try:
        print("🔄 Loading ML model...")
        detector = CICDAnomalyDetector()
        model_path = 'cicd_anomaly_model.joblib'
        
        if os.path.exists(model_path):
            detector.load_model(model_path)
            model_loaded = True
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ Model file not found. Training new model...")
            # Train a new model if none exists
            detector.train('large_logs_dataset.csv')
            detector.save_model(model_path)
            model_loaded = True
            print("✅ New model trained and loaded!")
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(traceback.format_exc())
        model_loaded = False

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'ok',
        'service': 'ml-anomaly-detection',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict anomaly for a given log message
    
    Expected payload:
    {
        "log_message": "Build failed with error code 1"
    }
    
    Returns:
    {
        "prediction": "normal" | "anomaly",
        "confidence": 0.85,
        "anomaly_score": -0.23
    }
    """
    try:
        # Check if model is loaded
        if not model_loaded or detector is None:
            return jsonify({
                'error': 'Model not loaded',
                'prediction': 'normal',
                'confidence': 0.5
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'log_message' not in data:
            return jsonify({
                'error': 'log_message is required',
                'prediction': 'normal',
                'confidence': 0.5
            }), 400
        
        log_message = data['log_message']
        
        if not log_message or not isinstance(log_message, str):
            return jsonify({
                'error': 'Invalid log_message',
                'prediction': 'normal',
                'confidence': 0.5
            }), 400
        
        # Make prediction
        result = detector.predict_issue(log_message)
        
        # Add request metadata
        result['service'] = 'ml-anomaly-detection'
        result['model_version'] = '1.0'
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'prediction': 'normal',
            'confidence': 0.5
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict anomalies for multiple log messages
    
    Expected payload:
    {
        "log_messages": [
            "Build started successfully",
            "Error: Module not found"
        ]
    }
    
    Returns:
    {
        "predictions": [
            {"log_message": "...", "prediction": "normal", "confidence": 0.9},
            {"log_message": "...", "prediction": "anomaly", "confidence": 0.8}
        ]
    }
    """
    try:
        # Check if model is loaded
        if not model_loaded or detector is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'log_messages' not in data:
            return jsonify({
                'error': 'log_messages array is required'
            }), 400
        
        log_messages = data['log_messages']
        
        if not isinstance(log_messages, list):
            return jsonify({
                'error': 'log_messages must be an array'
            }), 400
        
        # Limit batch size
        if len(log_messages) > 100:
            return jsonify({
                'error': 'Batch size limited to 100 messages'
            }), 400
        
        # Make predictions
        predictions = []
        for log_message in log_messages:
            if isinstance(log_message, str) and log_message.strip():
                result = detector.predict_issue(log_message)
                result['log_message'] = log_message[:100] + '...' if len(log_message) > 100 else log_message
                predictions.append(result)
            else:
                predictions.append({
                    'log_message': str(log_message)[:100],
                    'prediction': 'normal',
                    'confidence': 0.5,
                    'error': 'Invalid log message'
                })
        
        return jsonify({
            'predictions': predictions,
            'total_processed': len(predictions),
            'service': 'ml-anomaly-detection'
        })
        
    except Exception as e:
        print(f"❌ Error during batch prediction: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'error': 'Batch prediction failed',
            'details': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model
    """
    try:
        if not model_loaded or detector is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        # Try to get model information
        info = {
            'model_type': 'IsolationForest',
            'feature_extraction': 'TF-IDF',
            'model_loaded': model_loaded,
            'preprocessing': {
                'max_features': 1000,
                'ngram_range': [1, 2],
                'contamination': 0.2
            }
        }
        
        # Check if model files exist
        model_file = 'cicd_anomaly_model.joblib'
        if os.path.exists(model_file):
            info['model_file'] = model_file
            info['model_file_size'] = os.path.getsize(model_file)
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get model info',
            'details': str(e)
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain the model with new data (if provided)
    """
    try:
        global detector, model_loaded
        
        print("🔄 Retraining model...")
        
        # Initialize new detector
        detector = CICDAnomalyDetector()
        
        # Train with existing data (could be extended to accept new training data)
        detector.train('logs.csv')
        detector.save_model('cicd_anomaly_model.joblib')
        
        model_loaded = True
        
        print("✅ Model retrained successfully!")
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'model_version': '1.0'
        })
        
    except Exception as e:
        print(f"❌ Error retraining model: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Model retraining failed',
            'details': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'POST /predict': 'Predict anomaly for single log message',
            'POST /batch_predict': 'Predict anomalies for multiple log messages',
            'GET /health': 'Service health check',
            'GET /model/info': 'Model information',
            'POST /retrain': 'Retrain the model'
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting ML Anomaly Detection Service")
    print("=" * 50)
    
    # Load model on startup
    load_model()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 Service starting on port {port}")
    print(f"🔗 Health check: http://localhost:{port}/health")
    print(f"🤖 Prediction API: http://localhost:{port}/predict")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
