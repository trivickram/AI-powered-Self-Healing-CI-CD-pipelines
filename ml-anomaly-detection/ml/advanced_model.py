#!/usr/bin/env python3
"""
Enhanced CI/CD Anomaly Detection Model
Optimized for large datasets with advanced feature engineering and multiple algorithms
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedCICDAnomalyDetector:
    def __init__(self, model_type='isolation_forest'):
        """
        Initialize the enhanced anomaly detector
        
        Args:
            model_type: 'isolation_forest', 'one_class_svm', or 'hybrid'
        """
        self.model_type = model_type
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.label_encoders = {}
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.feature_importance = None
        self.performance_metrics = {}
        
        print(f"🤖 Initializing Advanced CI/CD Anomaly Detector")
        print(f"📊 Model Type: {model_type}")
    
    def _extract_text_features(self, df):
        """Extract advanced text features from log messages"""
        print("🔤 Extracting text features...")
        
        # Basic text statistics
        df['message_length'] = df['log_message'].str.len()
        df['word_count'] = df['log_message'].str.split().str.len()
        df['char_count'] = df['log_message'].str.count(r'\w')
        df['uppercase_ratio'] = df['log_message'].str.count(r'[A-Z]') / df['message_length']
        df['digit_ratio'] = df['log_message'].str.count(r'\d') / df['message_length']
        df['special_char_ratio'] = df['log_message'].str.count(r'[^\w\s]') / df['message_length']
        
        # Keyword presence features
        error_keywords = ['error', 'failed', 'failure', 'exception', 'fatal', 'critical']
        warning_keywords = ['warning', 'warn', 'deprecated', 'timeout']
        success_keywords = ['success', 'completed', 'passed', 'ok', 'healthy']
        performance_keywords = ['slow', 'fast', 'memory', 'cpu', 'disk', 'network']
        
        df['error_keyword_count'] = df['log_message'].str.lower().str.count('|'.join(error_keywords))
        df['warning_keyword_count'] = df['log_message'].str.lower().str.count('|'.join(warning_keywords))
        df['success_keyword_count'] = df['log_message'].str.lower().str.count('|'.join(success_keywords))
        df['performance_keyword_count'] = df['log_message'].str.lower().str.count('|'.join(performance_keywords))
        
        # Pattern-based features
        df['contains_ip'] = df['log_message'].str.contains(r'\d+\.\d+\.\d+\.\d+', regex=True).astype(int)
        df['contains_url'] = df['log_message'].str.contains(r'http[s]?://', regex=True).astype(int)
        df['contains_path'] = df['log_message'].str.contains(r'[/\\]', regex=True).astype(int)
        df['contains_version'] = df['log_message'].str.contains(r'v?\d+\.\d+', regex=True).astype(int)
        df['contains_code'] = df['log_message'].str.contains(r'exit code|error code|status code', regex=True).astype(int)
        
        return df
    
    def _engineer_temporal_features(self, df):
        """Engineer temporal features from timestamps"""
        print("⏰ Engineering temporal features...")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def _engineer_categorical_features(self, df):
        """Engineer features from categorical columns"""
        print("🏷️ Engineering categorical features...")
        
        categorical_cols = ['service', 'environment', 'severity', 'component']
        
        for col in categorical_cols:
            if col in df.columns:
                # Encode categorical variables
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories during prediction
                    known_categories = set(self.label_encoders[col].classes_)
                    df[f'{col}_temp'] = df[col].astype(str).apply(
                        lambda x: x if x in known_categories else 'unknown'
                    )
                    if 'unknown' not in known_categories:
                        # Add unknown category
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[f'{col}_temp'])
                    df.drop(f'{col}_temp', axis=1, inplace=True)
        
        return df
    
    def _engineer_numerical_features(self, df):
        """Engineer features from numerical columns"""
        print("📊 Engineering numerical features...")
        
        if 'duration_ms' in df.columns:
            df['duration_log'] = np.log1p(df['duration_ms'])
            df['duration_category'] = pd.cut(df['duration_ms'], 
                                           bins=[0, 1000, 5000, 15000, float('inf')], 
                                           labels=['fast', 'medium', 'slow', 'very_slow'])
            df['duration_category_encoded'] = LabelEncoder().fit_transform(df['duration_category'])
        
        return df
    
    def prepare_features(self, df):
        """Comprehensive feature preparation pipeline"""
        print("🔧 Preparing features...")
        
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Extract various types of features
        df_processed = self._extract_text_features(df_processed)
        df_processed = self._engineer_temporal_features(df_processed)
        df_processed = self._engineer_categorical_features(df_processed)
        df_processed = self._engineer_numerical_features(df_processed)
        
        # TF-IDF vectorization for text
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True,
                strip_accents='ascii',
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(df_processed['log_message'])
        else:
            tfidf_features = self.tfidf_vectorizer.transform(df_processed['log_message'])
        
        # Create TF-IDF feature DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Select numerical features for scaling
        numerical_features = [
            'message_length', 'word_count', 'char_count', 'uppercase_ratio',
            'digit_ratio', 'special_char_ratio', 'error_keyword_count',
            'warning_keyword_count', 'success_keyword_count', 'performance_keyword_count',
            'contains_ip', 'contains_url', 'contains_path', 'contains_version', 'contains_code'
        ]
        
        # Add temporal features if available
        if 'hour' in df_processed.columns:
            numerical_features.extend(['hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_night'])
        
        # Add categorical encoded features
        categorical_encoded_features = [col for col in df_processed.columns if col.endswith('_encoded')]
        numerical_features.extend(categorical_encoded_features)
        
        # Add duration features if available
        if 'duration_log' in df_processed.columns:
            numerical_features.extend(['duration_log', 'duration_category_encoded'])
        
        # Filter features that actually exist in the dataframe
        existing_features = [f for f in numerical_features if f in df_processed.columns]
        
        # Create feature matrix
        feature_df = df_processed[existing_features].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Scale numerical features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            scaled_features = self.feature_scaler.fit_transform(feature_df)
        else:
            scaled_features = self.feature_scaler.transform(feature_df)
        
        # Combine TF-IDF and numerical features
        final_features = np.hstack([tfidf_features.toarray(), scaled_features])
        
        self.feature_columns = list(tfidf_df.columns) + existing_features
        
        return final_features, df_processed
    
    def train(self, csv_file, test_size=0.2):
        """Train the anomaly detection model with advanced features"""
        print("🚀 Starting advanced model training...")
        print("=" * 50)
        
        # Load data
        print(f"📊 Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df):,} log entries")
        
        # Prepare features
        X, df_processed = self.prepare_features(df)
        print(f"📈 Feature matrix shape: {X.shape}")
        
        # Prepare labels
        y = (df['status'] == 'anomaly').astype(int)
        
        print(f"📊 Class distribution:")
        print(f"   Normal: {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
        print(f"   Anomaly: {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model based on type
        if self.model_type == 'isolation_forest':
            self._train_isolation_forest(X_train, y_train, X_test, y_test)
        elif self.model_type == 'one_class_svm':
            self._train_one_class_svm(X_train, y_train, X_test, y_test)
        elif self.model_type == 'hybrid':
            self._train_hybrid_model(X_train, y_train, X_test, y_test)
        
        self.is_trained = True
        print("✅ Advanced model training completed successfully!")
        
        return self.performance_metrics
    
    def _train_isolation_forest(self, X_train, y_train, X_test, y_test):
        """Train Isolation Forest with hyperparameter tuning"""
        print("🌲 Training Isolation Forest...")
        
        # Hyperparameter tuning
        param_grid = {
            'contamination': [0.1, 0.15, 0.2, 0.25],
            'n_estimators': [100, 200, 300],
            'max_samples': ['auto', 0.5, 0.8],
            'random_state': [42]
        }
        
        best_score = -float('inf')
        best_params = None
        
        for contamination in param_grid['contamination']:
            for n_estimators in param_grid['n_estimators']:
                for max_samples in param_grid['max_samples']:
                    model = IsolationForest(
                        contamination=contamination,
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    model.fit(X_train)
                    y_pred = model.predict(X_test)
                    y_pred_binary = (y_pred == -1).astype(int)
                    
                    # Use F1 score for evaluation
                    from sklearn.metrics import f1_score
                    score = f1_score(y_test, y_pred_binary)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'contamination': contamination,
                            'n_estimators': n_estimators,
                            'max_samples': max_samples
                        }
        
        print(f"🎯 Best parameters: {best_params}")
        print(f"📊 Best F1 score: {best_score:.3f}")
        
        # Train final model with best parameters
        self.model = IsolationForest(
            contamination=best_params['contamination'],
            n_estimators=best_params['n_estimators'],
            max_samples=best_params['max_samples'],
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train)
        self._evaluate_model(X_test, y_test)
    
    def _train_one_class_svm(self, X_train, y_train, X_test, y_test):
        """Train One-Class SVM"""
        print("🎯 Training One-Class SVM...")
        
        # Use only normal samples for training
        X_train_normal = X_train[y_train == 0]
        
        # Hyperparameter tuning
        param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'nu': [0.1, 0.15, 0.2, 0.25]
        }
        
        best_score = -float('inf')
        best_params = None
        
        for kernel in param_grid['kernel']:
            for gamma in param_grid['gamma']:
                for nu in param_grid['nu']:
                    try:
                        model = OneClassSVM(
                            kernel=kernel,
                            gamma=gamma,
                            nu=nu
                        )
                        
                        model.fit(X_train_normal)
                        y_pred = model.predict(X_test)
                        y_pred_binary = (y_pred == -1).astype(int)
                        
                        from sklearn.metrics import f1_score
                        score = f1_score(y_test, y_pred_binary)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'kernel': kernel,
                                'gamma': gamma,
                                'nu': nu
                            }
                    except:
                        continue
        
        print(f"🎯 Best parameters: {best_params}")
        print(f"📊 Best F1 score: {best_score:.3f}")
        
        # Train final model
        self.model = OneClassSVM(
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            nu=best_params['nu']
        )
        
        self.model.fit(X_train_normal)
        self._evaluate_model(X_test, y_test)
    
    def _train_hybrid_model(self, X_train, y_train, X_test, y_test):
        """Train hybrid ensemble model"""
        print("🔄 Training Hybrid Ensemble Model...")
        
        # Train multiple models
        models = {}
        
        # Isolation Forest
        if_model = IsolationForest(contamination=0.18, n_estimators=200, random_state=42, n_jobs=-1)
        if_model.fit(X_train)
        models['isolation_forest'] = if_model
        
        # One-Class SVM with normal samples
        X_train_normal = X_train[y_train == 0]
        svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.18)
        svm_model.fit(X_train_normal)
        models['one_class_svm'] = svm_model
        
        # Random Forest for comparison
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        self.model = models
        self._evaluate_hybrid_model(X_test, y_test)
    
    def _evaluate_model(self, X_test, y_test):
        """Evaluate single model performance"""
        print("📊 Evaluating model performance...")
        
        # Make predictions
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X_test)
            if self.model_type in ['isolation_forest', 'one_class_svm']:
                y_pred_binary = (y_pred == -1).astype(int)
            else:
                y_pred_binary = y_pred
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        
        self.performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_size': len(y_test)
        }
        
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1 Score:  {f1:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        print(f"\n📊 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0,0]:,}")
        print(f"   False Positives: {cm[0,1]:,}")
        print(f"   False Negatives: {cm[1,0]:,}")
        print(f"   True Positives:  {cm[1,1]:,}")
    
    def _evaluate_hybrid_model(self, X_test, y_test):
        """Evaluate hybrid model performance"""
        print("📊 Evaluating hybrid model performance...")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.model.items():
            if name == 'random_forest':
                pred = model.predict(X_test)
                predictions[name] = pred
            else:
                pred = model.predict(X_test)
                pred_binary = (pred == -1).astype(int)
                predictions[name] = pred_binary
        
        # Ensemble prediction (majority vote)
        ensemble_pred = np.array([
            predictions['isolation_forest'],
            predictions['one_class_svm'],
            predictions['random_forest']
        ])
        
        # Majority vote
        final_pred = (np.sum(ensemble_pred, axis=0) >= 2).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, final_pred)
        precision = precision_score(y_test, final_pred, zero_division=0)
        recall = recall_score(y_test, final_pred, zero_division=0)
        f1 = f1_score(y_test, final_pred, zero_division=0)
        
        self.performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'individual_models': {}
        }
        
        # Individual model performance
        for name, pred in predictions.items():
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1_ind = f1_score(y_test, pred, zero_division=0)
            
            self.performance_metrics['individual_models'][name] = {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1_ind
            }
        
        print(f"\n📈 Ensemble Model Performance:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1 Score:  {f1:.3f}")
        
        print(f"\n📊 Individual Model Performance:")
        for name, metrics in self.performance_metrics['individual_models'].items():
            print(f"   {name}: F1={metrics['f1_score']:.3f}, Acc={metrics['accuracy']:.3f}")
    
    def predict_issue(self, log_message):
        """Predict if a log message indicates an anomaly"""
        if not self.is_trained:
            return {
                'prediction': 'normal',
                'confidence': 0.5,
                'anomaly_score': 0.0,
                'error': 'Model not trained'
            }
        
        # Create DataFrame for feature engineering
        df = pd.DataFrame({'log_message': [log_message]})
        
        # Add default values for missing columns
        for col in ['service', 'environment', 'severity', 'component', 'duration_ms']:
            if col not in df.columns:
                df[col] = 'unknown' if col != 'duration_ms' else 1000
        
        # Prepare features
        try:
            X, _ = self.prepare_features(df)
            
            if self.model_type == 'hybrid':
                # Ensemble prediction
                predictions = []
                for name, model in self.model.items():
                    if name == 'random_forest':
                        pred = model.predict(X)[0]
                        predictions.append(pred)
                    else:
                        pred = model.predict(X)[0]
                        pred_binary = 1 if pred == -1 else 0
                        predictions.append(pred_binary)
                
                # Majority vote
                final_pred = 1 if sum(predictions) >= 2 else 0
                confidence = sum(predictions) / len(predictions)
                
            else:
                # Single model prediction
                pred = self.model.predict(X)[0]
                
                if self.model_type in ['isolation_forest', 'one_class_svm']:
                    final_pred = 1 if pred == -1 else 0
                    
                    # Get anomaly score for confidence
                    if hasattr(self.model, 'score_samples'):
                        score = self.model.score_samples(X)[0]
                        confidence = abs(score) / 2  # Normalize to 0-1 range
                    else:
                        confidence = 0.8 if final_pred == 1 else 0.9
                else:
                    final_pred = pred
                    confidence = 0.8 if final_pred == 1 else 0.9
            
            return {
                'prediction': 'anomaly' if final_pred == 1 else 'normal',
                'confidence': min(max(confidence, 0.0), 1.0),
                'anomaly_score': float(final_pred),
                'model_type': self.model_type
            }
            
        except Exception as e:
            return {
                'prediction': 'normal',
                'confidence': 0.5,
                'anomaly_score': 0.0,
                'error': str(e)
            }
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_scaler': self.feature_scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.feature_scaler = model_data['feature_scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Model loaded from {filepath}")

def main():
    """Main function to train and test the enhanced model"""
    print("🤖 Advanced CI/CD Anomaly Detection Model")
    print("=" * 60)
    
    # Test different model types
    model_types = ['isolation_forest', 'one_class_svm', 'hybrid']
    results = {}
    
    for model_type in model_types:
        print(f"\n🚀 Training {model_type} model...")
        print("-" * 40)
        
        detector = AdvancedCICDAnomalyDetector(model_type=model_type)
        
        # Train on large dataset
        metrics = detector.train('large_logs_dataset.csv')
        results[model_type] = metrics
        
        # Save model
        detector.save_model(f'advanced_{model_type}_model.joblib')
        
        # Test prediction
        test_logs = [
            "Build completed successfully in 45 seconds",
            "ERROR: Build failed with exit code 1",
            "CRITICAL: Out of memory during compilation",
            "Tests passed with 94% coverage"
        ]
        
        print(f"\n🧪 Testing {model_type} predictions:")
        for log in test_logs:
            result = detector.predict_issue(log)
            print(f"   '{log[:50]}...' → {result['prediction']} (conf: {result['confidence']:.2f})")
    
    # Compare results
    print(f"\n📊 Model Comparison:")
    print("-" * 60)
    for model_type, metrics in results.items():
        if 'f1_score' in metrics:
            print(f"{model_type:20}: F1={metrics['f1_score']:.3f}, Acc={metrics['accuracy']:.3f}")
        else:
            print(f"{model_type:20}: Training completed")

if __name__ == "__main__":
    main()
