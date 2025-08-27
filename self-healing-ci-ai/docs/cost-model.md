# 💰 Cost Model & Free Tier Optimization

## Overview

This project is specifically designed to operate within AWS Free Tier limits for educational and demonstration purposes. With proper configuration, monthly costs should remain under **$5** for typical usage patterns.

## AWS Free Tier Breakdown

### 🆓 Always Free Services

| Service | Free Tier Allowance | Project Usage | Status |
|---------|-------------------|---------------|---------|
| **AWS Lambda** | 1M requests/month + 400,000 GB-seconds compute | ~100 invocations/month | ✅ Free |
| **Amazon S3** | 5GB standard storage + 20,000 GET requests | <1GB (30-day lifecycle) | ✅ Free |
| **Amazon ECR** | 500MB storage/month (private repositories) | ~100MB for container images | ✅ Free |
| **CloudWatch Logs** | 5GB data ingestion + 5GB storage | ~2GB/month (14-day retention) | ✅ Free |
| **CloudWatch Metrics** | 10 custom metrics + 1M API requests | Standard AWS metrics only | ✅ Free |

### ⏰ 12-Month Free Tier

| Service | Free Tier Allowance | Project Usage | Status |
|---------|-------------------|---------------|---------|
| **Amazon ECS** | 20GB-hours Fargate compute/month | ~15GB-hours (single task) | ✅ Free |

### 💵 Pay-Per-Use Services

| Service | Pricing | Project Usage | Monthly Cost |
|---------|---------|---------------|--------------|
| **Amazon SES** | $0.10 per 1,000 emails | ~50 emails/month | **$0.005** |
| **Data Transfer** | $0.09/GB out (first 1GB free) | <1GB/month | **$0.00** |

## 📊 Usage Patterns & Projections

### Light Usage (Student/Demo)
- **CI/CD Runs**: 20-30 per month
- **Failed Builds**: 5-10 per month (triggering AI analysis)
- **Successful Deployments**: 15-20 per month
- **Email Notifications**: 30-50 per month
- **Total Monthly Cost**: **$0.01 - $0.50**

### Medium Usage (Small Team)
- **CI/CD Runs**: 100-200 per month
- **Failed Builds**: 20-40 per month
- **Successful Deployments**: 80-160 per month
- **Email Notifications**: 120-240 per month
- **Total Monthly Cost**: **$1.00 - $3.00**

### Heavy Usage (Active Development)
- **CI/CD Runs**: 500+ per month
- **Failed Builds**: 100+ per month
- **Successful Deployments**: 400+ per month
- **Email Notifications**: 600+ per month
- **Total Monthly Cost**: **$5.00 - $15.00**

## 🔧 Cost Optimization Strategies

### 1. Log Management
```terraform
# S3 Lifecycle Policy (in infra/s3.tf)
resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  rule {
    id     = "delete_old_logs"
    status = "Enabled"
    
    expiration {
      days = 30  # Reduce to 7 days for aggressive cost savings
    }
  }
}
```

### 2. CloudWatch Retention
```terraform
# CloudWatch Log Groups (in infra/cloudwatch.tf)
resource "aws_cloudwatch_log_group" "lambda" {
  retention_in_days = 14  # Reduce to 7 days if needed
}
```

### 3. Lambda Optimization
```terraform
# Lambda Configuration (in infra/lambda.tf)
resource "aws_lambda_function" "ai_analyzer" {
  memory_size = 256   # Increase to 512MB for faster execution (lower duration costs)
  timeout     = 30    # Reduce if analysis completes faster
}
```

### 4. ECS Right-Sizing
```terraform
# ECS Task Definition (in infra/ecr_ecs.tf)
resource "aws_ecs_task_definition" "app" {
  cpu    = 256   # 0.25 vCPU (minimum for Fargate)
  memory = 512   # 512MB (minimum for our app)
}
```

## 📈 Scaling Beyond Free Tier

### Cost Projections at Scale

| Monthly Usage | Lambda | S3 | ECS | ECR | SES | Total |
|---------------|--------|----|----|-----|-----|-------|
| **1,000 runs** | $0.20 | $0.10 | $12.00 | $0.10 | $0.10 | **$12.50** |
| **5,000 runs** | $1.00 | $0.25 | $12.00 | $0.20 | $0.50 | **$13.95** |
| **10,000 runs** | $2.00 | $0.50 | $24.00 | $0.30 | $1.00 | **$27.80** |

### Cost-Saving Alternatives for Scale

#### 1. Switch to Lambda-Only Hosting
Replace ECS with Lambda for application hosting:
```yaml
# Alternative deployment in GitHub Actions
- name: Deploy to Lambda
  run: |
    aws lambda update-function-code \
      --function-name my-app \
      --zip-file fileb://app.zip
```
**Savings**: Eliminates ECS costs (~$12/month)

#### 2. Use CloudWatch Events Instead of Polling
```terraform
# Event-driven architecture
resource "aws_cloudwatch_event_rule" "build_failure" {
  event_pattern = jsonencode({
    source = ["aws.codebuild"]
    detail = {
      build-status = ["FAILED"]
    }
  })
}
```

#### 3. Optimize LLM Usage
- **Use Groq** (default): $0.10-0.30 per 1M tokens
- **Batch multiple failures**: Analyze multiple logs in one call
- **Cache common solutions**: Store frequent fixes in DynamoDB

## 🚨 Cost Monitoring & Alerts

### 1. AWS Billing Alerts
```terraform
resource "aws_cloudwatch_metric_alarm" "billing_alarm" {
  alarm_name          = "billing-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"
  statistic           = "Maximum"
  threshold           = "10.00"  # $10 monthly limit
  alarm_description   = "Billing alert"
  
  dimensions = {
    Currency = "USD"
  }
}
```

### 2. Service-Specific Monitors
```bash
# Check S3 storage usage
aws s3api list-objects-v2 --bucket $BUCKET_NAME --query 'sum(Contents[].Size)'

# Check Lambda invocation count
aws logs filter-log-events \
  --log-group-name /aws/lambda/self-healing-ci-ai-analyzer \
  --start-time $(date -d '1 month ago' +%s)000 \
  --filter-pattern "START RequestId"
```

## 🎯 Free Tier Optimization Checklist

### ✅ Setup Phase
- [ ] Use `t2.micro` or `t3.micro` for any EC2 instances (if needed)
- [ ] Set S3 lifecycle policies to delete objects after 30 days
- [ ] Configure CloudWatch log retention to 14 days or less
- [ ] Use minimal ECS task sizes (0.25 vCPU, 512MB memory)
- [ ] Enable S3 server-side encryption (no additional cost)

### ✅ Runtime Phase
- [ ] Monitor Lambda execution duration (optimize for speed)
- [ ] Batch API calls to reduce request counts
- [ ] Use efficient LLM prompts to minimize token usage
- [ ] Clean up unused ECR images regularly
- [ ] Archive old logs instead of deleting (if needed for compliance)

### ✅ Monitoring Phase
- [ ] Set up billing alerts at $5, $10, and $20
- [ ] Review AWS Cost Explorer monthly
- [ ] Monitor CloudWatch metrics for usage patterns
- [ ] Optimize based on actual usage data

## 💡 Pro Tips for Cost Management

### 1. Development vs Production
- **Development**: Use minimal resources, shorter retention periods
- **Production**: Right-size based on actual load, implement auto-scaling

### 2. Regional Considerations
- **ap-south-1** (Mumbai): Generally cost-effective for Indian users
- **Data transfer**: Keep all resources in the same region
- **SES pricing**: Varies by region, check current rates

### 3. Alternative Architectures
For ultra-low cost, consider:
- **Serverless-only**: Lambda + API Gateway + DynamoDB
- **Event-driven**: CloudWatch Events + SNS + SQS
- **Static hosting**: S3 + CloudFront for frontend dashboards

## 📞 Cost Escalation Scenarios

### When Free Tier Expires (12 months)
- **ECS costs** become primary expense (~$12-30/month)
- **Alternative**: Migrate to Lambda-based hosting
- **Budget**: Plan for $20-50/month for production workloads

### Unexpected Usage Spikes
- **Lambda timeouts**: Increase memory instead of timeout duration
- **S3 costs**: Check for log retention issues or unexpected writes
- **ECS costs**: Verify auto-scaling policies aren't over-provisioning

### Getting Help
- **AWS Support**: Basic support included with free tier
- **AWS Cost Explorer**: Analyze spending patterns
- **AWS Trusted Advisor**: Recommendations for cost optimization
- **Community**: Stack Overflow, AWS forums for optimization tips
