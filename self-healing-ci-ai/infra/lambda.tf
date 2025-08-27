# Create Lambda deployment package
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda_ai_analyzer"
  output_path = "${path.module}/lambda_function.zip"
  excludes    = ["__pycache__", "*.pyc", ".pytest_cache", "tests"]
}

# Lambda function
resource "aws_lambda_function" "ai_analyzer" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.project_prefix}-ai-analyzer"
  role            = aws_iam_role.lambda_role.arn
  handler         = "main.lambda_handler"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  runtime         = "python3.11"
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory

  environment {
    variables = {
      AWS_REGION       = var.aws_region
      LOG_BUCKET       = aws_s3_bucket.logs.id
      SES_FROM_EMAIL   = var.ses_from_email
      SES_TO_EMAIL     = var.ses_to_email
      GITHUB_TOKEN     = var.github_token
      LLM_API_KEY      = var.llm_api_key
      PROVIDER         = var.llm_provider
      LOG_LEVEL        = "INFO"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic,
    aws_cloudwatch_log_group.lambda,
  ]

  tags = merge(local.common_tags, {
    Name = "${var.project_prefix}-ai-analyzer"
  })
}

# Lambda function alias for stable invocation
resource "aws_lambda_alias" "ai_analyzer" {
  name             = "production"
  description      = "Production alias for AI analyzer"
  function_name    = aws_lambda_function.ai_analyzer.function_name
  function_version = "$LATEST"
}

# Lambda permission for manual invocation (if needed)
resource "aws_lambda_permission" "allow_manual_invoke" {
  statement_id  = "AllowManualInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ai_analyzer.function_name
  principal     = "arn:aws:iam::${local.account_id}:root"
}
