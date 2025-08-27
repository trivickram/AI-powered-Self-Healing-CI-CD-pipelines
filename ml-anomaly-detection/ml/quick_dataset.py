import pandas as pd
import random
from datetime import datetime

# Simple test to generate a larger dataset
print("🚀 Generating large CI/CD dataset...")

# Basic templates
normal_logs = [
    "Build completed successfully",
    "Tests passed with 95% coverage", 
    "Deployment to staging successful",
    "Health check passed",
    "Dependencies installed successfully",
    "Code compilation successful",
    "Database migration completed",
    "Service started successfully",
    "Load balancer health check passed",
    "Metrics collection enabled"
]

anomaly_logs = [
    "ERROR: Build failed with exit code 1",
    "FATAL: Out of memory during compilation", 
    "CRITICAL: Database connection failed",
    "ERROR: Test timeout after 30 minutes",
    "FATAL: Deployment failed - service unreachable",
    "ERROR: Network connectivity issues",
    "CRITICAL: High CPU usage detected: 95%",
    "ERROR: Container failed to start",
    "FATAL: Authentication service unavailable", 
    "ERROR: SSL certificate validation failed"
]

# Generate 5000 logs
logs = []
for i in range(5000):
    if i % 1000 == 0:
        print(f"Generated {i} logs...")
    
    if random.random() < 0.8:  # 80% normal
        log_message = random.choice(normal_logs)
        status = 'normal'
    else:  # 20% anomaly
        log_message = random.choice(anomaly_logs)
        status = 'anomaly'
    
    logs.append({
        'log_message': log_message,
        'status': status
    })

# Save to CSV
df = pd.DataFrame(logs)
df.to_csv('large_logs_dataset.csv', index=False)

print(f"✅ Generated {len(df)} logs")
print(f"Normal: {len(df[df['status'] == 'normal'])}")
print(f"Anomaly: {len(df[df['status'] == 'anomaly'])}")
print("💾 Saved to large_logs_dataset.csv")
