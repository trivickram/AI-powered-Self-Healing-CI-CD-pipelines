#!/usr/bin/env python3
"""
Large CI/CD Log Dataset Generator
Generates thousands of realistic CI/CD logs for training robust ML models
"""

import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import json
import os

class CICDLogGenerator:
    def __init__(self):
        # Normal log templates
        self.normal_templates = [
            # Build phase
            "Starting build process for commit {commit_hash}",
            "Initializing build environment",
            "Installing dependencies from package.json",
            "Installing dependencies from requirements.txt",
            "Dependencies installed successfully in {duration}s",
            "Running npm install",
            "Running pip install -r requirements.txt",
            "Build completed successfully in {duration}s",
            "Compiling TypeScript files",
            "Compilation successful",
            "Webpack build completed",
            "Assets bundled successfully",
            "Docker image built successfully",
            "Image tagged as {image_tag}",
            
            # Test phase
            "Starting test suite execution",
            "Running unit tests",
            "Running integration tests",
            "Running end-to-end tests",
            "Test suite completed: {passed} passed, {failed} failed",
            "All tests passed successfully",
            "Code coverage: {coverage}%",
            "Test execution completed in {duration}s",
            "Linting completed with no issues",
            "Security scan completed - no vulnerabilities found",
            "Performance tests passed",
            "Load testing completed successfully",
            
            # Deploy phase
            "Starting deployment to {environment}",
            "Deploying to staging environment",
            "Deploying to production environment",
            "Health check passed",
            "Application started successfully",
            "Database migration completed",
            "Cache warmed up successfully",
            "Load balancer health check passed",
            "Deployment completed successfully",
            "Service is healthy and responding",
            "Rollout completed: 100% traffic routed",
            
            # Monitoring
            "CPU usage: {cpu}%",
            "Memory usage: {memory}%",
            "Response time: {response_time}ms",
            "Throughput: {throughput} req/s",
            "Error rate: {error_rate}%",
            "Service mesh health check passed",
            "Metrics collection enabled",
            "Logging configured successfully",
            
            # Git operations
            "Checked out branch {branch}",
            "Merged pull request #{pr_number}",
            "Commit {commit_hash} validated",
            "Code quality gate passed",
            "Branch protection rules satisfied",
            "Webhook triggered successfully",
            
            # Container operations
            "Container started successfully",
            "Pod status: Running",
            "Service endpoints updated",
            "Ingress rules applied",
            "Volume mounted successfully",
            "Secrets loaded successfully",
            
            # Database operations
            "Database connection established",
            "Schema validation passed",
            "Data backup completed",
            "Indexes optimized",
            "Query performance within limits",
            "Transaction completed successfully",
            
            # Notifications
            "Slack notification sent",
            "Email report generated",
            "Teams webhook delivered",
            "Status badge updated",
            "Dashboard metrics refreshed",
        ]
        
        # Anomaly log templates
        self.anomaly_templates = [
            # Build failures
            "ERROR: npm install failed with exit code 1",
            "ERROR: pip install failed - package not found",
            "FATAL: Out of memory during compilation",
            "ERROR: Module '{module}' not found",
            "BUILD FAILED: Compilation errors detected",
            "ERROR: Docker build failed - base image not found",
            "FATAL: Webpack build crashed with OOM error",
            "ERROR: TypeScript compilation failed",
            "CRITICAL: Build timeout after {timeout} minutes",
            "ERROR: Dependency conflict detected",
            "FATAL: Disk space insufficient for build",
            "ERROR: Build tool version mismatch",
            "CRITICAL: License check failed - invalid license",
            "ERROR: Code signing failed - certificate expired",
            "FATAL: Build environment corrupted",
            
            # Test failures
            "TEST FAILED: {test_count} tests failed",
            "ERROR: Test timeout after {timeout} minutes",
            "CRITICAL: Code coverage below threshold: {coverage}%",
            "ERROR: Integration test database connection failed",
            "FATAL: End-to-end test environment unreachable",
            "ERROR: Memory leak detected in tests",
            "CRITICAL: Security vulnerability found",
            "ERROR: Performance test failed - response time {time}ms",
            "FATAL: Load test crashed the system",
            "ERROR: Flaky test detected - intermittent failures",
            "CRITICAL: Regression detected in test suite",
            "ERROR: Mock service unavailable",
            "FATAL: Test data corruption detected",
            
            # Deployment failures
            "DEPLOY FAILED: Unable to connect to {environment}",
            "ERROR: Database migration failed",
            "CRITICAL: Health check failed after deployment",
            "ERROR: Container failed to start",
            "FATAL: Service discovery registration failed",
            "ERROR: Load balancer health check timeout",
            "CRITICAL: Rollback failed - system in inconsistent state",
            "ERROR: Configuration validation failed",
            "FATAL: SSL certificate validation failed",
            "ERROR: Network connectivity issues detected",
            "CRITICAL: Deployment timeout after {timeout} minutes",
            "ERROR: Resource limits exceeded",
            "FATAL: Blue-green deployment failed",
            "ERROR: Canary deployment showing errors",
            
            # Infrastructure failures
            "CRITICAL: High CPU usage detected: {cpu}%",
            "ERROR: Memory usage critical: {memory}%",
            "FATAL: Disk space critical: {disk}% remaining",
            "ERROR: Network latency spike: {latency}ms",
            "CRITICAL: Service mesh failure detected",
            "ERROR: Database connection pool exhausted",
            "FATAL: Cache cluster failure",
            "ERROR: Message queue overflow",
            "CRITICAL: Storage backend failure",
            "ERROR: CDN origin server unreachable",
            
            # Security issues
            "SECURITY ALERT: Unauthorized access attempt",
            "CRITICAL: API rate limit exceeded",
            "ERROR: Authentication service unavailable",
            "FATAL: Data breach attempt detected",
            "SECURITY: Suspicious activity in logs",
            "CRITICAL: Malware signature detected",
            "ERROR: Certificate authority unreachable",
            "FATAL: Encryption key rotation failed",
            "SECURITY: Privilege escalation attempt",
            "CRITICAL: DDoS attack detected",
            
            # Application errors
            "APPLICATION ERROR: Unhandled exception",
            "FATAL: Database deadlock detected",
            "ERROR: External API service unavailable",
            "CRITICAL: Payment gateway failure",
            "ERROR: Session store corruption",
            "FATAL: Message broker connection lost",
            "APPLICATION ERROR: Invalid configuration",
            "CRITICAL: Data consistency violation",
            "ERROR: Worker process crashed",
            "FATAL: Scheduler service failure",
            
            # Environment issues
            "ENV ERROR: Environment variable missing: {var_name}",
            "CRITICAL: Configuration file corrupted",
            "ERROR: Service dependency unavailable",
            "FATAL: Kubernetes cluster unreachable",
            "ERROR: Container registry authentication failed",
            "CRITICAL: Helm deployment failed",
            "ERROR: Terraform apply failed",
            "FATAL: Cloud provider API rate limited",
            "ERROR: DNS resolution failure",
            "CRITICAL: VPN connection lost",
        ]
        
        # Additional context generators
        self.environments = ["dev", "staging", "production", "test", "qa", "sandbox"]
        self.branches = ["main", "master", "develop", "feature/auth", "feature/api", "hotfix/urgent", "release/v1.2.0"]
        self.services = ["api", "frontend", "backend", "database", "cache", "worker", "scheduler", "auth"]
        self.technologies = ["node.js", "python", "java", "docker", "kubernetes", "terraform", "react", "angular"]
        self.error_codes = [1, 2, 125, 126, 127, 128, 130, 137, 139, 143]
        
    def generate_normal_log(self):
        """Generate a normal (non-anomalous) log entry"""
        template = random.choice(self.normal_templates)
        
        # Fill in template variables
        variables = {
            'commit_hash': self._generate_commit_hash(),
            'duration': random.randint(30, 600),
            'image_tag': f"v{random.randint(1,10)}.{random.randint(0,20)}.{random.randint(0,50)}",
            'passed': random.randint(50, 200),
            'failed': 0,
            'coverage': random.randint(80, 98),
            'environment': random.choice(self.environments),
            'cpu': random.randint(20, 70),
            'memory': random.randint(30, 75),
            'response_time': random.randint(50, 200),
            'throughput': random.randint(100, 1000),
            'error_rate': round(random.uniform(0, 2), 2),
            'branch': random.choice(self.branches),
            'pr_number': random.randint(1, 1000),
        }
        
        try:
            return template.format(**variables)
        except KeyError:
            return template
    
    def generate_anomaly_log(self):
        """Generate an anomalous log entry"""
        template = random.choice(self.anomaly_templates)
        
        # Fill in template variables
        variables = {
            'module': random.choice(['pandas', 'numpy', 'express', 'react', 'lodash']),
            'timeout': random.randint(30, 120),
            'coverage': random.randint(10, 65),
            'test_count': random.randint(1, 50),
            'time': random.randint(5000, 30000),
            'environment': random.choice(self.environments),
            'cpu': random.randint(85, 100),
            'memory': random.randint(90, 100),
            'disk': random.randint(1, 10),
            'latency': random.randint(2000, 10000),
            'var_name': random.choice(['DATABASE_URL', 'API_KEY', 'SECRET_KEY']),
        }
        
        try:
            return template.format(**variables)
        except KeyError:
            return template
    
    def _generate_commit_hash(self):
        """Generate a realistic git commit hash"""
        chars = 'abcdef0123456789'
        return ''.join(random.choice(chars) for _ in range(7))
    
    def _generate_timestamp(self, base_time, variation_hours=24):
        """Generate a realistic timestamp"""
        delta = timedelta(hours=random.randint(-variation_hours, variation_hours))
        return base_time + delta
    
    def generate_dataset(self, total_logs=10000, anomaly_ratio=0.15):
        """
        Generate a large dataset of CI/CD logs
        
        Args:
            total_logs: Total number of logs to generate
            anomaly_ratio: Proportion of logs that should be anomalies (0.15 = 15%)
        """
        print(f"🚀 Generating {total_logs:,} CI/CD logs...")
        print(f"📊 Anomaly ratio: {anomaly_ratio:.1%}")
        
        logs = []
        base_time = datetime.now()
        
        # Calculate counts
        anomaly_count = int(total_logs * anomaly_ratio)
        normal_count = total_logs - anomaly_count
        
        print(f"✅ Normal logs: {normal_count:,}")
        print(f"❌ Anomaly logs: {anomaly_count:,}")
        
        # Generate normal logs
        print("📝 Generating normal logs...")
        for i in range(normal_count):
            if i % 1000 == 0:
                print(f"   Progress: {i:,}/{normal_count:,}")
            
            log_entry = {
                'timestamp': self._generate_timestamp(base_time).isoformat(),
                'log_message': self.generate_normal_log(),
                'status': 'normal',
                'service': random.choice(self.services),
                'environment': random.choice(self.environments),
                'severity': random.choice(['INFO', 'DEBUG', 'TRACE']),
                'component': random.choice(['build', 'test', 'deploy', 'monitor']),
                'duration_ms': random.randint(100, 5000),
                'success': True
            }
            logs.append(log_entry)
        
        # Generate anomaly logs
        print("⚠️ Generating anomaly logs...")
        for i in range(anomaly_count):
            if i % 1000 == 0:
                print(f"   Progress: {i:,}/{anomaly_count:,}")
            
            log_entry = {
                'timestamp': self._generate_timestamp(base_time).isoformat(),
                'log_message': self.generate_anomaly_log(),
                'status': 'anomaly',
                'service': random.choice(self.services),
                'environment': random.choice(self.environments),
                'severity': random.choice(['ERROR', 'CRITICAL', 'FATAL', 'WARN']),
                'component': random.choice(['build', 'test', 'deploy', 'monitor']),
                'duration_ms': random.randint(1000, 30000),
                'success': False
            }
            logs.append(log_entry)
        
        # Shuffle the logs to mix normal and anomaly entries
        print("🔀 Shuffling logs...")
        random.shuffle(logs)
        
        return logs
    
    def save_to_csv(self, logs, filename='large_logs_dataset.csv'):
        """Save logs to CSV file"""
        print(f"💾 Saving {len(logs):,} logs to {filename}...")
        
        df = pd.DataFrame(logs)
        df.to_csv(filename, index=False)
        
        print(f"✅ Dataset saved successfully!")
        print(f"📁 File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        
        # Print dataset statistics
        print("\n📊 Dataset Statistics:")
        print(f"   Total logs: {len(df):,}")
        print(f"   Normal logs: {len(df[df['status'] == 'normal']):,} ({len(df[df['status'] == 'normal'])/len(df)*100:.1f}%)")
        print(f"   Anomaly logs: {len(df[df['status'] == 'anomaly']):,} ({len(df[df['status'] == 'anomaly'])/len(df)*100:.1f}%)")
        print(f"   Unique services: {df['service'].nunique()}")
        print(f"   Unique environments: {df['environment'].nunique()}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def generate_additional_features(self, logs):
        """Add additional engineered features for better ML performance"""
        print("🔧 Engineering additional features...")
        
        for log in logs:
            # Message length
            log['message_length'] = len(log['log_message'])
            
            # Word count
            log['word_count'] = len(log['log_message'].split())
            
            # Contains specific keywords
            log['contains_error'] = any(word in log['log_message'].lower() for word in ['error', 'failed', 'failure'])
            log['contains_warning'] = any(word in log['log_message'].lower() for word in ['warning', 'warn'])
            log['contains_timeout'] = any(word in log['log_message'].lower() for word in ['timeout', 'timed out'])
            log['contains_memory'] = any(word in log['log_message'].lower() for word in ['memory', 'oom', 'out of memory'])
            log['contains_network'] = any(word in log['log_message'].lower() for word in ['network', 'connection', 'connectivity'])
            
            # Time of day (could affect system performance)
            timestamp = datetime.fromisoformat(log['timestamp'])
            log['hour_of_day'] = timestamp.hour
            log['day_of_week'] = timestamp.weekday()
            log['is_weekend'] = timestamp.weekday() >= 5
            log['is_business_hours'] = 9 <= timestamp.hour <= 17
            
            # Duration categories
            if log['duration_ms'] < 1000:
                log['duration_category'] = 'fast'
            elif log['duration_ms'] < 5000:
                log['duration_category'] = 'medium'
            else:
                log['duration_category'] = 'slow'
        
        return logs

def main():
    """Main function to generate the large dataset"""
    generator = CICDLogGenerator()
    
    # Configuration
    TOTAL_LOGS = 25000  # 25K logs for robust training
    ANOMALY_RATIO = 0.18  # 18% anomalies (realistic for CI/CD)
    
    print("🤖 Large CI/CD Log Dataset Generator")
    print("=" * 50)
    
    # Generate logs
    logs = generator.generate_dataset(TOTAL_LOGS, ANOMALY_RATIO)
    
    # Add engineered features
    logs = generator.generate_additional_features(logs)
    
    # Save to CSV
    df = generator.save_to_csv(logs, 'large_logs_dataset.csv')
    
    # Create a smaller sample for quick testing
    print("\n🧪 Creating test sample...")
    sample_df = df.sample(n=1000, random_state=42)
    sample_df.to_csv('sample_logs_dataset.csv', index=False)
    print(f"✅ Test sample saved: sample_logs_dataset.csv (1,000 logs)")
    
    # Validation
    print("\n🔍 Dataset Validation:")
    print(f"   ✅ No missing values: {df.isnull().sum().sum() == 0}")
    print(f"   ✅ Balanced classes: {df['status'].value_counts()}")
    print(f"   ✅ Diverse messages: {df['log_message'].nunique():,} unique messages")
    print(f"   ✅ Multiple environments: {list(df['environment'].unique())}")
    print(f"   ✅ Various severities: {list(df['severity'].unique())}")
    
    print("\n🎉 Large dataset generation completed successfully!")
    print(f"📈 Ready for robust ML model training with {TOTAL_LOGS:,} logs")

if __name__ == "__main__":
    main()
