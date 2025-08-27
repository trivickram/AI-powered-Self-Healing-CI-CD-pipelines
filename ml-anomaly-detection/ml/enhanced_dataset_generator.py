#!/usr/bin/env python3
"""
Enhanced Dataset Generator for 95%+ Accuracy
Creates more sophisticated and realistic CI/CD anomaly patterns
"""

import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import re

class EnhancedDatasetGenerator:
    def __init__(self):
        # More sophisticated normal patterns with realistic variations
        self.normal_patterns = [
            # Build Success Patterns
            "Build #{build_num} completed successfully in {duration}s for commit {commit}",
            "Maven build successful: {artifact} version {version} compiled in {duration}s",
            "Gradle build finished: All {module_count} modules built successfully",
            "Docker image {image}:{tag} built and pushed to registry in {duration}s",
            "NPM install completed: {package_count} packages installed in {duration}s",
            "Compilation successful: {file_count} files processed, 0 errors, 0 warnings",
            "TypeScript compilation completed: Generated {output_count} output files",
            "Webpack bundling successful: {bundle_size}MB bundle created in {duration}s",
            
            # Test Success Patterns  
            "Test suite passed: {passed} tests, 0 failures, 0 skipped in {duration}s",
            "Unit tests completed: {test_count} tests passed, coverage: {coverage}%",
            "Integration tests successful: All {endpoint_count} endpoints validated",
            "E2E tests passed: {scenario_count} scenarios executed successfully",
            "Security scan completed: No vulnerabilities detected in {scan_duration}s",
            "Code quality gate passed: Technical debt ratio: {debt_ratio}%",
            "Performance tests passed: Response time: {response_time}ms (target: <500ms)",
            "Load testing successful: {rps} RPS sustained for {duration} minutes",
            
            # Deploy Success Patterns
            "Deployment to {environment} successful: Version {version} deployed in {duration}s",
            "Rolling update completed: {instances} instances updated to version {version}",
            "Blue-green deployment successful: Traffic switched to green environment",
            "Canary deployment validated: 5% traffic shows healthy metrics",
            "Database migration completed: {migration_count} migrations applied successfully",
            "Configuration update deployed: {config_count} config files updated",
            "SSL certificate renewal successful: Valid until {expiry_date}",
            "CDN cache invalidation completed: {path_count} paths invalidated",
            
            # Infrastructure Success
            "Health check passed: All {service_count} services responding normally",
            "Auto-scaling event: Scaled up to {instance_count} instances due to high load",
            "Backup completed: {backup_size}GB backed up to {storage_location}",
            "Log rotation successful: {log_count} log files archived",
            "Monitoring alert resolved: CPU usage normalized to {cpu_usage}%",
            "Network connectivity verified: All {endpoint_count} endpoints reachable"
        ]
        
        # Highly sophisticated anomaly patterns with real error signatures
        self.anomaly_patterns = [
            # Build Failures with specific error codes
            "BUILD FAILED: Exit code 1 - Compilation error in {file}:{line}",
            "FATAL: Out of memory error during compilation (heap size: {heap_size}MB)",
            "ERROR: Maven dependency resolution failed - {artifact} not found in repository",
            "CRITICAL: Docker build failed - Base image {image} not found or access denied",
            "BUILD ERROR: NPM install failed - Package {package} has security vulnerabilities",
            "COMPILATION FAILED: TypeScript error TS{error_code} in {file}:{line}:{column}",
            "FATAL: Gradle build timeout after {timeout} minutes - Build process hung",
            "ERROR: Webpack build failed - Module {module} not found",
            "CRITICAL: Build disk space insufficient - Required: {required}GB, Available: {available}GB",
            "BUILD FAILED: License validation error - Prohibited license detected in {package}",
            
            # Test Failures with detailed diagnostics
            "TEST FAILED: {failed_count} of {total_count} tests failed in {test_suite}",
            "ASSERTION ERROR: Expected {expected}, got {actual} in test {test_name}",
            "CRITICAL: Test timeout - {test_name} exceeded {timeout}s limit",
            "FLAKY TEST DETECTED: {test_name} failed {failure_count} times in last {runs} runs",
            "INTEGRATION TEST FAILED: Service {service} returned HTTP {status_code}",
            "E2E TEST ERROR: Element '{selector}' not found on page {page_url}",
            "SECURITY VULNERABILITY: {severity} severity issue found in {component}",
            "CODE COVERAGE BELOW THRESHOLD: {actual_coverage}% < {required_coverage}%",
            "PERFORMANCE TEST FAILED: Response time {actual_time}ms > {max_time}ms",
            "LOAD TEST FAILED: System crashed at {rps} RPS after {duration} seconds",
            
            # Deployment Failures with recovery info
            "DEPLOYMENT FAILED: Rollback initiated to version {previous_version}",
            "CRITICAL: Database migration failed at step {step} - Data inconsistency detected",
            "DEPLOY ERROR: Health check failed for {service} - Service not responding",
            "ROLLBACK FAILED: Cannot revert to previous version - Manual intervention required",
            "CONFIGURATION ERROR: Invalid config value for {parameter} in {environment}",
            "DEPLOYMENT TIMEOUT: {service} failed to start within {timeout} seconds",
            "CRITICAL: Blue-green deployment failed - Green environment health check failed",
            "CANARY DEPLOYMENT ABORTED: Error rate {error_rate}% exceeds threshold {threshold}%",
            "SSL CERTIFICATE ERROR: Certificate validation failed for {domain}",
            "CDN DEPLOYMENT FAILED: Origin server {server} unreachable",
            
            # Infrastructure Failures with system details
            "SYSTEM CRITICAL: Memory usage {memory_usage}% exceeds threshold {threshold}%",
            "NETWORK ERROR: Connection timeout to {service} after {timeout}s",
            "DATABASE CRITICAL: Connection pool exhausted - {active}/{max} connections",
            "DISK SPACE CRITICAL: {filesystem} at {usage}% capacity ({free}GB remaining)",
            "SERVICE UNAVAILABLE: {service} health check failed - Response time: {time}ms",
            "AUTHENTICATION FAILED: Unable to connect to {service} - Invalid credentials",
            "RATE LIMIT EXCEEDED: API {api} hit rate limit {limit} requests/minute",
            "CACHE FAILURE: Redis cluster down - Failover to backup cache failed",
            "MESSAGE QUEUE OVERFLOW: {queue_name} has {message_count} pending messages",
            "MONITORING ALERT: {metric} value {value} exceeds critical threshold {threshold}",
            
            # Application Runtime Errors
            "APPLICATION ERROR: Unhandled exception {exception_type} in {method}",
            "FATAL: Database deadlock detected - Transaction rolled back after {timeout}s",
            "CRITICAL: Memory leak detected in {component} - Heap usage growing continuously",
            "ERROR: External API {api} returning HTTP {status_code} - Service degraded",
            "WORKER PROCESS CRASHED: PID {pid} terminated with signal {signal}",
            "SESSION STORE CORRUPTION: Unable to deserialize session data for user {user_id}",
            "PAYMENT GATEWAY ERROR: Transaction {transaction_id} failed with code {error_code}",
            "DATA CORRUPTION DETECTED: Checksum mismatch in file {filename}",
            "SCHEDULER FAILURE: Cron job {job_name} failed to execute at {scheduled_time}",
            "CIRCUIT BREAKER OPEN: {service} marked as unavailable after {failure_count} failures"
        ]
        
        # Enhanced context generators for realism
        self.build_numbers = lambda: random.randint(1000, 9999)
        self.durations = lambda: random.choice([
            random.randint(30, 120),    # Fast builds
            random.randint(121, 300),   # Medium builds  
            random.randint(301, 600),   # Slow builds
            random.randint(601, 1800)   # Very slow builds
        ])
        self.commit_hashes = lambda: ''.join(random.choices('abcdef0123456789', k=7))
        self.versions = lambda: f"{random.randint(1,5)}.{random.randint(0,20)}.{random.randint(0,100)}"
        self.coverage_values = lambda: random.randint(70, 99)
        self.memory_values = lambda: random.randint(512, 8192)
        self.error_codes = lambda: random.choice([1, 2, 125, 126, 127, 128, 130, 137, 139, 143])
        self.http_codes = lambda: random.choice([400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504])
        
        # Realistic service and component names
        self.services = [
            'user-service', 'auth-service', 'payment-service', 'notification-service',
            'api-gateway', 'user-interface', 'database-service', 'cache-service',
            'message-queue', 'file-storage', 'search-service', 'analytics-service'
        ]
        
        self.environments = ['dev', 'test', 'staging', 'prod', 'qa', 'sandbox']
        self.severities = {
            'normal': ['INFO', 'DEBUG', 'TRACE'],
            'anomaly': ['ERROR', 'FATAL', 'CRITICAL', 'WARN']
        }
        
    def generate_enhanced_normal_log(self):
        """Generate a highly realistic normal log"""
        template = random.choice(self.normal_patterns)
        
        variables = {
            'build_num': self.build_numbers(),
            'duration': self.durations(),
            'commit': self.commit_hashes(),
            'version': self.versions(),
            'coverage': self.coverage_values(),
            'artifact': random.choice(['com.example.app', 'org.company.service', 'io.project.api']),
            'module_count': random.randint(3, 15),
            'package_count': random.randint(50, 500),
            'file_count': random.randint(100, 1000),
            'output_count': random.randint(50, 200),
            'bundle_size': round(random.uniform(0.5, 5.0), 1),
            'passed': random.randint(50, 500),
            'test_count': random.randint(100, 1000),
            'endpoint_count': random.randint(10, 50),
            'scenario_count': random.randint(5, 25),
            'scan_duration': random.randint(30, 300),
            'debt_ratio': round(random.uniform(1.0, 5.0), 1),
            'response_time': random.randint(50, 400),
            'rps': random.randint(100, 2000),
            'environment': random.choice(self.environments),
            'instances': random.randint(2, 20),
            'migration_count': random.randint(1, 10),
            'config_count': random.randint(5, 30),
            'expiry_date': (datetime.now() + timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
            'path_count': random.randint(10, 100),
            'service_count': random.randint(5, 20),
            'instance_count': random.randint(3, 15),
            'backup_size': round(random.uniform(1.0, 100.0), 1),
            'storage_location': random.choice(['S3', 'Azure Blob', 'GCS']),
            'log_count': random.randint(100, 1000),
            'cpu_usage': random.randint(20, 60),
            'image': random.choice(['nginx', 'alpine', 'ubuntu', 'node']),
            'tag': self.versions()
        }
        
        try:
            return template.format(**variables)
        except KeyError:
            return template
    
    def generate_enhanced_anomaly_log(self):
        """Generate a highly realistic anomaly log"""
        template = random.choice(self.anomaly_patterns)
        
        variables = {
            'file': random.choice(['App.java', 'main.py', 'server.js', 'config.xml', 'package.json']),
            'line': random.randint(1, 500),
            'column': random.randint(1, 80),
            'heap_size': random.choice([512, 1024, 2048, 4096, 8192]),
            'artifact': random.choice(['com.example:app:1.0', 'org.apache:commons:2.1', 'io.netty:core:4.1']),
            'package': random.choice(['lodash', 'moment', 'express', 'react', 'axios']),
            'error_code': random.choice([2307, 2322, 2345, 2571, 2739]),
            'timeout': random.choice([300, 600, 900, 1800, 3600]),
            'module': random.choice(['./src/utils', './components/Header', './services/api']),
            'required': round(random.uniform(5.0, 20.0), 1),
            'available': round(random.uniform(0.1, 3.0), 1),
            'failed_count': random.randint(1, 20),
            'total_count': random.randint(50, 200),
            'test_suite': random.choice(['UnitTests', 'IntegrationTests', 'E2ETests', 'ApiTests']),
            'expected': random.choice(['200', 'true', 'SUCCESS', 'null']),
            'actual': random.choice(['500', 'false', 'FAILED', 'undefined']),
            'test_name': random.choice(['testUserLogin', 'testPaymentProcess', 'testDataValidation']),
            'failure_count': random.randint(2, 10),
            'runs': random.randint(10, 50),
            'service': random.choice(self.services),
            'status_code': self.http_codes(),
            'selector': random.choice(['#login-button', '.submit-form', '[data-test=checkout]']),
            'page_url': random.choice(['/login', '/checkout', '/dashboard', '/profile']),
            'severity': random.choice(['HIGH', 'MEDIUM', 'LOW', 'CRITICAL']),
            'component': random.choice(['authentication', 'payment', 'user-management', 'api']),
            'actual_coverage': random.randint(40, 75),
            'required_coverage': random.randint(80, 95),
            'actual_time': random.randint(1000, 10000),
            'max_time': random.choice([500, 1000, 2000, 5000]),
            'rps': random.randint(500, 5000),
            'previous_version': self.versions(),
            'step': random.randint(1, 20),
            'parameter': random.choice(['database_url', 'api_key', 'max_connections']),
            'environment': random.choice(self.environments),
            'error_rate': round(random.uniform(5.0, 25.0), 1),
            'threshold': random.choice([1.0, 2.0, 5.0, 10.0]),
            'domain': random.choice(['api.example.com', 'app.company.io', 'service.org']),
            'server': random.choice(['server-01', 'app-server-prod', 'api-gateway-1']),
            'memory_usage': random.randint(85, 99),
            'usage': random.randint(90, 99),
            'free': round(random.uniform(0.1, 2.0), 1),
            'filesystem': random.choice(['/var/log', '/tmp', '/opt/app']),
            'active': random.randint(95, 100),
            'max': 100,
            'time': random.randint(5000, 30000),
            'api': random.choice(['user-api', 'payment-api', 'auth-api']),
            'limit': random.choice([1000, 5000, 10000]),
            'queue_name': random.choice(['payment-queue', 'notification-queue', 'user-events']),
            'message_count': random.randint(10000, 100000),
            'metric': random.choice(['cpu_usage', 'memory_usage', 'error_rate', 'response_time']),
            'value': random.randint(90, 150),
            'exception_type': random.choice(['NullPointerException', 'SQLException', 'TimeoutException']),
            'method': random.choice(['processPayment()', 'getUserData()', 'validateInput()']),
            'pid': random.randint(1000, 9999),
            'signal': random.choice(['SIGKILL', 'SIGTERM', 'SIGSEGV']),
            'user_id': random.randint(1000, 999999),
            'transaction_id': f"TXN{random.randint(100000, 999999)}",
            'filename': random.choice(['data.db', 'users.json', 'config.xml']),
            'job_name': random.choice(['backup-job', 'cleanup-job', 'report-job']),
            'scheduled_time': datetime.now().strftime('%H:%M')
        }
        
        try:
            return template.format(**variables)
        except KeyError:
            return template
    
    def generate_enhanced_dataset(self, total_logs=30000, anomaly_ratio=0.18):
        """Generate enhanced dataset for 95%+ accuracy"""
        print(f"🚀 Generating enhanced dataset with {total_logs:,} logs...")
        print(f"📊 Target anomaly ratio: {anomaly_ratio:.1%}")
        
        logs = []
        anomaly_count = int(total_logs * anomaly_ratio)
        normal_count = total_logs - anomaly_count
        
        print(f"✅ Normal logs: {normal_count:,}")
        print(f"❌ Anomaly logs: {anomaly_count:,}")
        
        # Generate normal logs
        print("📝 Generating enhanced normal logs...")
        for i in range(normal_count):
            if i % 2000 == 0:
                print(f"   Progress: {i:,}/{normal_count:,}")
            
            log_entry = {
                'timestamp': self._generate_timestamp().isoformat(),
                'log_message': self.generate_enhanced_normal_log(),
                'status': 'normal',
                'service': random.choice(self.services),
                'environment': random.choice(self.environments),
                'severity': random.choice(self.severities['normal']),
                'component': random.choice(['build', 'test', 'deploy', 'monitor', 'infrastructure']),
                'duration_ms': random.randint(100, 5000),
                'success': True
            }
            logs.append(log_entry)
        
        # Generate anomaly logs  
        print("⚠️ Generating enhanced anomaly logs...")
        for i in range(anomaly_count):
            if i % 1000 == 0:
                print(f"   Progress: {i:,}/{anomaly_count:,}")
            
            log_entry = {
                'timestamp': self._generate_timestamp().isoformat(),
                'log_message': self.generate_enhanced_anomaly_log(),
                'status': 'anomaly',
                'service': random.choice(self.services),
                'environment': random.choice(self.environments),
                'severity': random.choice(self.severities['anomaly']),
                'component': random.choice(['build', 'test', 'deploy', 'monitor', 'infrastructure']),
                'duration_ms': random.randint(1000, 60000),
                'success': False
            }
            logs.append(log_entry)
        
        # Shuffle logs
        print("🔀 Shuffling logs...")
        random.shuffle(logs)
        
        return logs
    
    def _generate_timestamp(self):
        """Generate realistic timestamp"""
        base_time = datetime.now()
        delta_hours = random.randint(-48, 0)  # Last 48 hours
        delta_minutes = random.randint(0, 59)
        delta_seconds = random.randint(0, 59)
        
        return base_time + timedelta(hours=delta_hours, minutes=delta_minutes, seconds=delta_seconds)
    
    def save_enhanced_dataset(self, logs, filename='enhanced_large_dataset.csv'):
        """Save enhanced dataset"""
        print(f"💾 Saving enhanced dataset to {filename}...")
        
        df = pd.DataFrame(logs)
        df.to_csv(filename, index=False)
        
        print(f"✅ Enhanced dataset saved!")
        print(f"📁 File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        
        # Statistics
        print(f"\n📊 Enhanced Dataset Statistics:")
        print(f"   Total logs: {len(df):,}")
        print(f"   Normal: {len(df[df['status'] == 'normal']):,} ({len(df[df['status'] == 'normal'])/len(df)*100:.1f}%)")
        print(f"   Anomaly: {len(df[df['status'] == 'anomaly']):,} ({len(df[df['status'] == 'anomaly'])/len(df)*100:.1f}%)")
        print(f"   Unique messages: {df['log_message'].nunique():,}")
        print(f"   Services: {list(df['service'].unique())}")
        print(f"   Environments: {list(df['environment'].unique())}")
        print(f"   Severities: {list(df['severity'].unique())}")
        
        return df

def main():
    """Generate enhanced dataset for 95%+ accuracy"""
    print("🎯 Enhanced Dataset Generator for 95%+ Accuracy")
    print("=" * 60)
    
    generator = EnhancedDatasetGenerator()
    
    # Generate 30K logs with more sophisticated patterns
    logs = generator.generate_enhanced_dataset(total_logs=30000, anomaly_ratio=0.18)
    
    # Save dataset
    df = generator.save_enhanced_dataset(logs)
    
    print(f"\n🎉 Enhanced dataset generation completed!")
    print(f"📈 Ready for ultra high-performance training with sophisticated patterns")

if __name__ == "__main__":
    import os
    main()
