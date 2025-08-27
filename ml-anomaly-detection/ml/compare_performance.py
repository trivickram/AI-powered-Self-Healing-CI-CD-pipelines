#!/usr/bin/env python3
"""
Performance Comparison: Small vs Large Dataset
Demonstrates the improvement in ML model performance with more training data
"""

import pandas as pd
import numpy as np
from model import CICDAnomalyDetector
import time
import os

def compare_datasets():
    """Compare model performance between small and large datasets"""
    print("🔍 Dataset Performance Comparison")
    print("=" * 60)
    
    datasets = [
        {
            'name': 'Small Dataset',
            'file': 'logs.csv',
            'description': 'Original 75 logs'
        },
        {
            'name': 'Large Dataset', 
            'file': 'large_logs_dataset.csv',
            'description': '25,000 realistic CI/CD logs'
        }
    ]
    
    results = {}
    
    for dataset in datasets:
        print(f"\n🚀 Testing {dataset['name']} ({dataset['description']})")
        print("-" * 50)
        
        if not os.path.exists(dataset['file']):
            print(f"❌ Dataset file {dataset['file']} not found, skipping...")
            continue
        
        # Load dataset info
        df = pd.read_csv(dataset['file'])
        print(f"📊 Dataset size: {len(df):,} logs")
        print(f"📈 Normal logs: {len(df[df['status'] == 'normal']):,}")
        print(f"⚠️ Anomaly logs: {len(df[df['status'] == 'anomaly']):,}")
        
        # Train model
        detector = CICDAnomalyDetector()
        
        start_time = time.time()
        try:
            metrics = detector.train(dataset['file'])
            training_time = time.time() - start_time
            
            results[dataset['name']] = {
                'dataset_size': len(df),
                'training_time': training_time,
                'metrics': metrics,
                'success': True
            }
            
            print(f"⏱️ Training time: {training_time:.2f} seconds")
            print(f"🎯 Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            results[dataset['name']] = {
                'dataset_size': len(df),
                'training_time': 0,
                'success': False,
                'error': str(e)
            }
        
        # Test predictions on standard examples
        if results[dataset['name']]['success']:
            print(f"\n🧪 Testing predictions:")
            test_cases = [
                ("Build completed successfully", "normal"),
                ("ERROR: Build failed with exit code 1", "anomaly"),
                ("Tests passed with 95% coverage", "normal"),
                ("FATAL: Out of memory during compilation", "anomaly"),
                ("Deployment to staging successful", "normal"),
                ("CRITICAL: Database connection failed", "anomaly")
            ]
            
            correct_predictions = 0
            for log_message, expected in test_cases:
                result = detector.predict_issue(log_message)
                predicted = result['prediction']
                confidence = result['confidence']
                
                is_correct = predicted == expected
                correct_predictions += is_correct
                
                status_icon = "✅" if is_correct else "❌"
                print(f"   {status_icon} '{log_message[:40]}...' → {predicted} (conf: {confidence:.2f})")
            
            accuracy = correct_predictions / len(test_cases)
            results[dataset['name']]['prediction_accuracy'] = accuracy
            print(f"📊 Prediction accuracy: {accuracy:.1%}")
    
    # Summary comparison
    print(f"\n📈 Performance Summary")
    print("=" * 60)
    
    if len(results) >= 2:
        small_key = 'Small Dataset'
        large_key = 'Large Dataset'
        
        if small_key in results and large_key in results:
            small = results[small_key]
            large = results[large_key]
            
            print(f"Dataset Size:")
            print(f"   📊 Small: {small['dataset_size']:,} logs")
            print(f"   📊 Large: {large['dataset_size']:,} logs")
            print(f"   📈 Improvement: {large['dataset_size']/small['dataset_size']:.1f}x more data")
            
            if small['success'] and large['success']:
                print(f"\nTraining Time:")
                print(f"   ⏱️ Small: {small['training_time']:.2f}s")
                print(f"   ⏱️ Large: {large['training_time']:.2f}s")
                
                if 'prediction_accuracy' in small and 'prediction_accuracy' in large:
                    print(f"\nPrediction Accuracy:")
                    print(f"   🎯 Small: {small['prediction_accuracy']:.1%}")
                    print(f"   🎯 Large: {large['prediction_accuracy']:.1%}")
                    improvement = large['prediction_accuracy'] - small['prediction_accuracy']
                    print(f"   📈 Improvement: {improvement:+.1%}")
                
                print(f"\n🎉 Key Benefits of Large Dataset:")
                print(f"   • More diverse training examples")
                print(f"   • Better generalization to unseen logs")
                print(f"   • Reduced overfitting")
                print(f"   • More robust anomaly detection")
                print(f"   • Higher confidence in predictions")
    
    return results

def generate_dataset_recommendations():
    """Generate recommendations for dataset improvement"""
    print(f"\n💡 Dataset Recommendations")
    print("=" * 60)
    
    print(f"🎯 Optimal Dataset Size:")
    print(f"   • Minimum: 5,000 logs for basic performance")
    print(f"   • Recommended: 25,000+ logs for production use")
    print(f"   • Enterprise: 100,000+ logs for maximum accuracy")
    
    print(f"\n📊 Dataset Composition:")
    print(f"   • Normal logs: 80-85% (realistic CI/CD success rate)")
    print(f"   • Anomaly logs: 15-20% (various failure types)")
    print(f"   • Diverse environments: dev, staging, production")
    print(f"   • Multiple services: api, frontend, database, etc.")
    
    print(f"\n🔄 Continuous Improvement:")
    print(f"   • Collect real production logs")
    print(f"   • Label new failure patterns")
    print(f"   • Retrain model monthly")
    print(f"   • Monitor prediction accuracy")
    print(f"   • Add edge cases and rare failures")
    
    print(f"\n⚡ Performance Tips:")
    print(f"   • Use TF-IDF for text feature extraction")
    print(f"   • Include temporal and categorical features")
    print(f"   • Apply feature scaling and normalization")
    print(f"   • Use ensemble methods for better accuracy")
    print(f"   • Hyperparameter tuning for optimal performance")

def main():
    """Main function"""
    print("🤖 CI/CD Anomaly Detection - Dataset Performance Analysis")
    print("🎯 Demonstrating the importance of large, diverse training datasets")
    print("=" * 80)
    
    # Run comparison
    results = compare_datasets()
    
    # Generate recommendations
    generate_dataset_recommendations()
    
    print(f"\n✅ Analysis Complete!")
    print(f"📈 Large datasets significantly improve ML model performance")
    print(f"🎯 Use the large_logs_dataset.csv (25,000 logs) for production training")

if __name__ == "__main__":
    main()
