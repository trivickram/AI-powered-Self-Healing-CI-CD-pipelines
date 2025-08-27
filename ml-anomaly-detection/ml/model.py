import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import re

class CICDAnomalyDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='ascii'
        )
        self.isolation_forest = IsolationForest(
            contamination=0.2,  # Expect 20% anomalies
            random_state=42,
            n_estimators=100
        )
        self.is_trained = False
    
    def preprocess_log_message(self, log_message):
        """
        Preprocess log message for better feature extraction
        """
        if not isinstance(log_message, str):
            return ""
        
        # Remove timestamps and numbers that might not be relevant
        log_message = re.sub(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', '', log_message)
        log_message = re.sub(r'\d+', 'NUM', log_message)
        
        # Remove file paths
        log_message = re.sub(r'[/\\][^\s]+', 'FILEPATH', log_message)
        
        # Normalize common error patterns
        log_message = re.sub(r'exit code \d+', 'exit code NUM', log_message)
        log_message = re.sub(r'port \d+', 'port NUM', log_message)
        
        return log_message.strip()
    
    def load_data(self, csv_path):
        """
        Load and preprocess training data from CSV
        """
        print(f"📊 Loading data from {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} log entries")
        
        # Preprocess log messages
        df['processed_log'] = df['log_message'].apply(self.preprocess_log_message)
        
        # Convert status to binary (normal=1, anomaly=-1 for IsolationForest)
        df['anomaly_label'] = df['status'].apply(lambda x: -1 if x == 'anomaly' else 1)
        
        return df
    
    def train(self, csv_file='large_logs_dataset.csv', test_size=0.2, contamination=0.18):
        """
        Train the anomaly detection model
        """
        print("🚀 Starting model training...")
        
        # Load data
        df = self.load_data(csv_file)
        
        # Vectorize log messages
        print("🔤 Vectorizing log messages...")
        X = self.vectorizer.fit_transform(df['processed_log'])
        y = df['anomaly_label']
        
        print(f"📈 Feature matrix shape: {X.shape}")
        print(f"📊 Class distribution:\n{df['status'].value_counts()}")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Isolation Forest
        print("🤖 Training Isolation Forest...")
        self.isolation_forest.fit(X_train)
        
        # Validate model
        print("📊 Evaluating model performance...")
        y_pred = self.isolation_forest.predict(X_test)
        
        # Convert predictions to match our labels (1=normal, -1=anomaly)
        y_pred_binary = ['normal' if pred == 1 else 'anomaly' for pred in y_pred]
        y_test_binary = ['normal' if label == 1 else 'anomaly' for label in y_test]
        
        print("\n📈 Classification Report:")
        print(classification_report(y_test_binary, y_pred_binary))
        
        print("\n📊 Confusion Matrix:")
        print(confusion_matrix(y_test_binary, y_pred_binary))
        
        # Calculate anomaly scores for confidence measurement
        scores = self.isolation_forest.decision_function(X_test)
        print(f"\n📊 Anomaly scores - Min: {scores.min():.3f}, Max: {scores.max():.3f}")
        
        self.is_trained = True
        print("✅ Model training completed successfully!")
        
        return {
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'accuracy': np.mean(y_pred == y_test),
            'score_range': (scores.min(), scores.max())
        }
    
    def predict_issue(self, log_message):
        """
        Predict if a log message indicates an anomaly
        
        Args:
            log_message (str): The log message to analyze
            
        Returns:
            dict: Prediction result with confidence score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not log_message or not isinstance(log_message, str):
            return {
                'prediction': 'normal',
                'confidence': 0.5,
                'error': 'Invalid or empty log message'
            }
        
        try:
            # Preprocess the log message
            processed_log = self.preprocess_log_message(log_message)
            
            # Vectorize
            X = self.vectorizer.transform([processed_log])
            
            # Predict
            prediction = self.isolation_forest.predict(X)[0]
            anomaly_score = self.isolation_forest.decision_function(X)[0]
            
            # Convert prediction to human-readable format
            result = 'normal' if prediction == 1 else 'anomaly'
            
            # Convert anomaly score to confidence (0-1 scale)
            # Negative scores indicate anomalies, positive indicate normal
            confidence = max(0.1, min(0.99, (anomaly_score + 0.5) / 1.0))
            if result == 'anomaly':
                confidence = 1 - confidence
            
            return {
                'prediction': result,
                'confidence': round(confidence, 3),
                'anomaly_score': round(anomaly_score, 3)
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return {
                'prediction': 'normal',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def save_model(self, model_path='cicd_anomaly_model.joblib'):
        """
        Save the trained model and vectorizer
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'isolation_forest': self.isolation_forest,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to {model_path}")
    
    def load_model(self, model_path='cicd_anomaly_model.joblib'):
        """
        Load a pre-trained model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.vectorizer = model_data['vectorizer']
        self.isolation_forest = model_data['isolation_forest']
        self.is_trained = model_data['is_trained']
        
        print(f"📥 Model loaded from {model_path}")

def main():
    """
    Main function to train and save the model
    """
    print("🚀 CI/CD Anomaly Detection Model Training")
    print("=" * 50)
    
    # Initialize detector
    detector = CICDAnomalyDetector()
    
    # Train model
    csv_path = 'logs.csv'
    training_results = detector.train(csv_path)
    
    print(f"\n📊 Training Results:")
    print(f"   Train samples: {training_results['train_size']}")
    print(f"   Test samples: {training_results['test_size']}")
    print(f"   Accuracy: {training_results['accuracy']:.3f}")
    
    # Save model
    model_path = 'cicd_anomaly_model.joblib'
    detector.save_model(model_path)
    
    # Test with sample predictions
    print(f"\n🧪 Testing predictions:")
    test_logs = [
        "Build started successfully",
        "All tests passed",
        "Error: Module not found",
        "npm ERR! Could not resolve dependency",
        "Connection timeout to database",
        "Segmentation fault occurred",
        "Deployment completed successfully",
        "Memory leak detected"
    ]
    
    for log in test_logs:
        result = detector.predict_issue(log)
        print(f"   '{log}' -> {result['prediction']} (confidence: {result['confidence']})")
    
    print(f"\n✅ Model training and testing completed!")
    print(f"📁 Model saved as: {model_path}")

if __name__ == "__main__":
    main()
