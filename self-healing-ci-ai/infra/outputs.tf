output "log_bucket_name" {
  description = "Name of the S3 bucket for logs"
  value       = aws_s3_bucket.logs.id
}

output "lambda_function_name" {
  description = "Name of the AI analyzer Lambda function"
  value       = aws_lambda_function.ai_analyzer.function_name
}

output "lambda_function_arn" {
  description = "ARN of the AI analyzer Lambda function"
  value       = aws_lambda_function.ai_analyzer.arn
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.app.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for Lambda"
  value       = aws_cloudwatch_log_group.lambda.name
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "setup_instructions" {
  description = "Setup instructions for GitHub Actions"
  value = <<-EOT
    
    🔧 SETUP INSTRUCTIONS:
    
    1. Add these secrets to your GitHub repository:
       - AWS_ACCESS_KEY_ID: (Your AWS access key)
       - AWS_SECRET_ACCESS_KEY: (Your AWS secret key)
       - AWS_REGION: ${var.aws_region}
       - ECR_REPO: ${split("/", aws_ecr_repository.app.repository_url)[1]}
       - ECS_CLUSTER: ${aws_ecs_cluster.main.name}
       - ECS_SERVICE: ${aws_ecs_service.app.name}
       - LOG_BUCKET: ${aws_s3_bucket.logs.id}
       - SES_FROM_EMAIL: ${var.ses_from_email}
       - SES_TO_EMAIL: ${var.ses_to_email}
       - LLM_API_KEY: (Your LLM provider API key)
       - GITHUB_TOKEN: (Your GitHub PAT)
    
    2. Verify your SES email address in AWS Console:
       https://console.aws.amazon.com/ses/home?region=${var.aws_region}#verified-senders-email:
    
    3. Set repository variable FAIL_TEST=1 to test the system:
       https://github.com/<YOUR_REPO>/settings/variables/actions
    
    4. Push a commit to trigger the pipeline!
    
  EOT
}
