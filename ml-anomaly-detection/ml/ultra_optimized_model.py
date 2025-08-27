#!/usr/bin/env python3
"""
Ultra-Optimized Model for 95%+ Accuracy
Combines multiple advanced ML techniques with sophisticated feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, SelectPercentile,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    LabelEncoder, OneHotEncoder, PowerTransformer
)
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import re
from collections import Counter
import dateutil.parser
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Advanced feature extractor for CI/CD logs"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for _, row in X.iterrows():
            message = str(row.get('log_message', ''))
            service = str(row.get('service', ''))
            environment = str(row.get('environment', ''))
            severity = str(row.get('severity', ''))
            component = str(row.get('component', ''))
            
            feature_dict = {}
            
            # Text-based features
            feature_dict.update(self._extract_text_features(message))
            feature_dict.update(self._extract_error_features(message))
            feature_dict.update(self._extract_temporal_features(row))
            feature_dict.update(self._extract_categorical_features(service, environment, severity, component))
            feature_dict.update(self._extract_numeric_features(message))
            feature_dict.update(self._extract_pattern_features(message))
            
            features.append(feature_dict)
        
        # Convert to DataFrame and fill missing values
        feature_df = pd.DataFrame(features).fillna(0)
        self.feature_names = feature_df.columns.tolist()
        
        return feature_df.values
    
    def _extract_text_features(self, message):
        """Extract comprehensive text features"""
        features = {}
        
        # Basic text statistics
        features['message_length'] = len(message)
        features['word_count'] = len(message.split())
        features['char_count'] = len(message)
        features['sentence_count'] = len(message.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in message.split()]) if message.split() else 0
        
        # Character analysis
        features['uppercase_ratio'] = sum(1 for c in message if c.isupper()) / len(message) if message else 0
        features['digit_ratio'] = sum(1 for c in message if c.isdigit()) / len(message) if message else 0
        features['special_char_ratio'] = sum(1 for c in message if not c.isalnum() and not c.isspace()) / len(message) if message else 0
        features['whitespace_ratio'] = sum(1 for c in message if c.isspace()) / len(message) if message else 0
        
        # Punctuation analysis
        punctuation_counts = Counter(c for c in message if c in '.,!?;:()[]{}')
        for punct in '.,!?;:()[]{}':
            features[f'punct_{punct}_count'] = punctuation_counts.get(punct, 0)
        
        return features
    
    def _extract_error_features(self, message):
        """Extract error-specific features"""
        features = {}
        message_lower = message.lower()
        
        # Error keywords
        error_keywords = [
            'error', 'failed', 'failure', 'exception', 'critical', 'fatal',
            'timeout', 'crash', 'abort', 'denied', 'refused', 'invalid',
            'corrupt', 'leak', 'overflow', 'underflow', 'deadlock',
            'unauthorized', 'forbidden', 'unavailable', 'unreachable'
        ]
        
        for keyword in error_keywords:
            features[f'has_{keyword}'] = int(keyword in message_lower)
        
        # Success keywords
        success_keywords = [
            'success', 'successful', 'completed', 'passed', 'deployed',
            'built', 'tested', 'validated', 'verified', 'healthy'
        ]
        
        for keyword in success_keywords:
            features[f'has_{keyword}'] = int(keyword in message_lower)
        
        # Error patterns
        features['has_error_code'] = int(bool(re.search(r'error\s*code?\s*:?\s*\d+', message_lower)))
        features['has_exit_code'] = int(bool(re.search(r'exit\s*code\s*:?\s*\d+', message_lower)))
        features['has_http_error'] = int(bool(re.search(r'http\s*[45]\d\d', message_lower)))
        features['has_stack_trace'] = int(bool(re.search(r'at\s+[\w.]+\([\w.]+:\d+\)', message)))
        features['has_exception_type'] = int(bool(re.search(r'\w+Exception|Error$', message)))
        
        return features
    
    def _extract_temporal_features(self, row):
        """Extract temporal features"""
        features = {}
        
        try:
            timestamp = dateutil.parser.parse(str(row.get('timestamp', '')))
            features['hour'] = timestamp.hour
            features['day_of_week'] = timestamp.weekday()
            features['is_weekend'] = int(timestamp.weekday() >= 5)
            features['is_business_hours'] = int(9 <= timestamp.hour <= 17)
            features['is_night'] = int(timestamp.hour < 6 or timestamp.hour > 22)
        except:
            features.update({
                'hour': 12, 'day_of_week': 1, 'is_weekend': 0,
                'is_business_hours': 1, 'is_night': 0
            })
        
        # Duration features
        duration = row.get('duration_ms', 0)
        features['duration_ms'] = duration
        features['is_long_duration'] = int(duration > 5000)
        features['is_very_long_duration'] = int(duration > 30000)
        features['duration_log'] = np.log1p(duration)
        
        return features
    
    def _extract_categorical_features(self, service, environment, severity, component):
        """Extract categorical features"""
        features = {}
        
        # Service features
        features[f'service_{service}'] = 1
        features['service_length'] = len(service)
        features['service_has_dash'] = int('-' in service)
        
        # Environment features
        features[f'env_{environment}'] = 1
        features['is_prod'] = int(environment == 'prod')
        features['is_test_env'] = int(environment in ['test', 'qa', 'staging'])
        
        # Severity features
        features[f'severity_{severity}'] = 1
        features['is_high_severity'] = int(severity in ['ERROR', 'FATAL', 'CRITICAL'])
        
        # Component features
        features[f'component_{component}'] = 1
        features['is_build_component'] = int(component == 'build')
        features['is_deploy_component'] = int(component == 'deploy')
        
        return features
    
    def _extract_numeric_features(self, message):
        """Extract numeric features from text"""
        features = {}
        
        # Find all numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        numbers = [float(n) for n in numbers]
        
        features['number_count'] = len(numbers)
        if numbers:
            features['max_number'] = max(numbers)
            features['min_number'] = min(numbers)
            features['avg_number'] = np.mean(numbers)
            features['sum_numbers'] = sum(numbers)
            features['has_large_number'] = int(any(n > 1000 for n in numbers))
        else:
            features.update({
                'max_number': 0, 'min_number': 0, 'avg_number': 0,
                'sum_numbers': 0, 'has_large_number': 0
            })
        
        # Specific numeric patterns
        features['has_version'] = int(bool(re.search(r'\d+\.\d+\.\d+', message)))
        features['has_timestamp'] = int(bool(re.search(r'\d{4}-\d{2}-\d{2}', message)))
        features['has_percentage'] = int('%' in message)
        features['has_file_size'] = int(bool(re.search(r'\d+\s*[kmg]b', message.lower())))
        
        return features
    
    def _extract_pattern_features(self, message):
        """Extract complex pattern features"""
        features = {}
        
        # File and path patterns
        features['has_file_path'] = int(bool(re.search(r'[/\\][\w./\\-]+', message)))
        features['has_url'] = int(bool(re.search(r'https?://[\w.-]+', message)))
        features['has_email'] = int(bool(re.search(r'\w+@\w+\.\w+', message)))
        features['has_ip_address'] = int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', message)))
        
        # Code patterns
        features['has_function_call'] = int(bool(re.search(r'\w+\([^)]*\)', message)))
        features['has_class_name'] = int(bool(re.search(r'[A-Z][a-zA-Z]*Exception|[A-Z][a-zA-Z]*Error', message)))
        features['has_variable'] = int(bool(re.search(r'\$\w+|{\w+}', message)))
        
        # Technical patterns
        features['has_hash'] = int(bool(re.search(r'\b[a-f0-9]{7,40}\b', message)))
        features['has_uuid'] = int(bool(re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', message)))
        features['has_json'] = int('{' in message and '}' in message)
        
        return features

class UltraOptimizedDetector:
    """Ultra-optimized anomaly detector for 95%+ accuracy"""
    
    def __init__(self):
        self.models = {}
        self.feature_extractor = AdvancedFeatureExtractor()
        self.text_vectorizers = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.best_features = None
        
    def prepare_advanced_features(self, df):
        """Prepare ultra-advanced feature set"""
        print("🔧 Preparing ultra-advanced features...")
        
        # Extract custom features
        custom_features = self.feature_extractor.fit_transform(df)
        custom_feature_names = self.feature_extractor.feature_names
        
        # Multiple text vectorizations
        print("📝 Creating multiple text vectorizations...")
        
        # TF-IDF with different configurations
        tfidf_configs = [
            {'max_features': 2000, 'ngram_range': (1, 1), 'min_df': 2, 'max_df': 0.95},
            {'max_features': 1500, 'ngram_range': (1, 2), 'min_df': 3, 'max_df': 0.9},
            {'max_features': 1000, 'ngram_range': (2, 3), 'min_df': 5, 'max_df': 0.8},
            {'max_features': 800, 'ngram_range': (1, 3), 'min_df': 2, 'max_df': 0.95, 'analyzer': 'char_wb'}
        ]
        
        text_features = []
        feature_names = []
        
        for i, config in enumerate(tfidf_configs):
            vectorizer = TfidfVectorizer(**config)
            tfidf_matrix = vectorizer.fit_transform(df['log_message'].fillna('')).toarray()
            self.text_vectorizers[f'tfidf_{i}'] = vectorizer
            text_features.append(tfidf_matrix)
            feature_names.extend([f'tfidf_{i}_{name}' for name in vectorizer.get_feature_names_out()])
        
        # Count vectorizer
        count_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), min_df=3)
        count_matrix = count_vectorizer.fit_transform(df['log_message'].fillna('')).toarray()
        self.text_vectorizers['count'] = count_vectorizer
        text_features.append(count_matrix)
        feature_names.extend([f'count_{name}' for name in count_vectorizer.get_feature_names_out()])
        
        # Combine all features
        all_features = np.hstack([custom_features] + text_features)
        all_feature_names = custom_feature_names + feature_names
        
        print(f"✅ Total features created: {all_features.shape[1]:,}")
        
        return all_features, all_feature_names
    
    def select_best_features(self, X, y, feature_names):
        """Select best features using multiple techniques"""
        print("🎯 Selecting best features...")
        
        # Remove low variance features
        var_selector = VarianceThreshold(threshold=0.001)
        X_var = var_selector.fit_transform(X)
        selected_features = np.array(feature_names)[var_selector.get_support()]
        
        print(f"   After variance threshold: {X_var.shape[1]:,} features")
        
        # Univariate feature selection
        k_best = SelectKBest(score_func=f_classif, k=min(3000, X_var.shape[1]))
        X_kbest = k_best.fit_transform(X_var, y)
        selected_features = selected_features[k_best.get_support()]
        
        print(f"   After K-best selection: {X_kbest.shape[1]:,} features")
        
        # Mutual information selection
        mi_selector = SelectPercentile(score_func=mutual_info_classif, percentile=80)
        X_mi = mi_selector.fit_transform(X_kbest, y)
        selected_features = selected_features[mi_selector.get_support()]
        
        print(f"   After mutual information: {X_mi.shape[1]:,} features")
        
        self.feature_selectors = {
            'variance': var_selector,
            'k_best': k_best,
            'mutual_info': mi_selector
        }
        
        self.best_features = selected_features
        return X_mi
    
    def create_ultra_ensemble(self):
        """Create ultra-sophisticated ensemble"""
        print("🤖 Creating ultra-sophisticated ensemble...")
        
        # Base models with optimized parameters
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                random_state=42
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(200, 100, 50), max_iter=500,
                random_state=42, early_stopping=True
            )),
            ('svm', SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True,
                random_state=42
            )),
            ('lr', LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, n_jobs=-1
            )),
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
            ('lda', LinearDiscriminantAnalysis()),
        ]
        
        # Create voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=-1
        )
        
        return voting_ensemble
    
    def train_ultra_model(self, df):
        """Train ultra-optimized model"""
        print("🚀 Training ultra-optimized model for 95%+ accuracy...")
        print("=" * 70)
        
        # Prepare features
        X, feature_names = self.prepare_advanced_features(df)
        y = (df['status'] == 'anomaly').astype(int)
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"   Normal samples: {(y == 0).sum():,}")
        print(f"   Anomaly samples: {(y == 1).sum():,}")
        print(f"   Anomaly ratio: {y.mean():.1%}")
        
        # Feature selection
        X_selected = self.select_best_features(X, y, feature_names)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("⚖️ Scaling features...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['robust'] = scaler
        
        # Create and train ensemble
        ensemble_model = self.create_ultra_ensemble()
        
        print("🎯 Training ultra-ensemble...")
        ensemble_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("📈 Evaluating model performance...")
        train_score = ensemble_model.score(X_train_scaled, y_train)
        test_score = ensemble_model.score(X_test_scaled, y_test)
        
        y_pred = ensemble_model.predict(X_test_scaled)
        y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate comprehensive metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print(f"\n🏆 ULTRA MODEL PERFORMANCE:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Avg Precision: {avg_precision:.4f}")
        print(f"   Train Score: {train_score:.4f}")
        print(f"   Test Score: {test_score:.4f}")
        
        # Cross-validation
        print("\n🔄 Cross-validation results:")
        cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store model
        self.models['ultra_ensemble'] = ensemble_model
        
        # Success indicator
        if accuracy >= 0.95:
            print(f"\n🎉 SUCCESS! Achieved {accuracy*100:.2f}% accuracy (>= 95% target)")
        elif accuracy >= 0.93:
            print(f"\n🎯 VERY CLOSE! Achieved {accuracy*100:.2f}% accuracy (target: 95%)")
        else:
            print(f"\n📈 GOOD PROGRESS! Achieved {accuracy*100:.2f}% accuracy (working toward 95%)")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'model': ensemble_model,
            'feature_count': X_selected.shape[1]
        }
    
    def predict(self, log_data):
        """Predict using ultra-optimized model"""
        if 'ultra_ensemble' not in self.models:
            raise ValueError("Model not trained. Call train_ultra_model first.")
        
        # Convert to DataFrame if needed
        if isinstance(log_data, dict):
            df = pd.DataFrame([log_data])
        else:
            df = log_data
        
        # Extract features
        X, _ = self.prepare_advanced_features(df)
        
        # Apply feature selection
        for selector_name, selector in self.feature_selectors.items():
            X = selector.transform(X)
        
        # Scale features
        X_scaled = self.scalers['robust'].transform(X)
        
        # Predict
        model = self.models['ultra_ensemble']
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities

def main():
    """Main training function"""
    print("🎯 Ultra-Optimized Anomaly Detection Model")
    print("Target: 95%+ Accuracy")
    print("=" * 60)
    
    # Try enhanced dataset first, fallback to original
    try:
        print("📁 Loading enhanced dataset...")
        df = pd.read_csv('enhanced_large_dataset.csv')
        print(f"✅ Enhanced dataset loaded: {len(df):,} logs")
    except FileNotFoundError:
        print("📁 Loading original large dataset...")
        df = pd.read_csv('large_logs_dataset.csv')
        print(f"✅ Original dataset loaded: {len(df):,} logs")
    
    # Create and train ultra model
    detector = UltraOptimizedDetector()
    results = detector.train_ultra_model(df)
    
    print(f"\n🎉 Ultra-optimized training completed!")
    print(f"📊 Final accuracy: {results['accuracy']*100:.2f}%")
    print(f"🔧 Features used: {results['feature_count']:,}")
    
    # Save model results
    import joblib
    print("💾 Saving ultra-optimized model...")
    joblib.dump(detector, 'ultra_optimized_model.pkl')
    print("✅ Model saved as 'ultra_optimized_model.pkl'")

if __name__ == "__main__":
    main()
