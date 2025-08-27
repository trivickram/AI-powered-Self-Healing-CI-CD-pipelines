variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "project_prefix" {
  description = "Prefix for all resource names"
  type        = string
  default     = "self-healing-ci"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "ses_from_email" {
  description = "SES verified sender email address"
  type        = string
}

variable "ses_to_email" {
  description = "Email address to receive notifications"
  type        = string
}

variable "github_token" {
  description = "GitHub personal access token"
  type        = string
  sensitive   = true
}

variable "llm_api_key" {
  description = "LLM provider API key"
  type        = string
  sensitive   = true
}

variable "llm_provider" {
  description = "LLM provider (groq, openai, huggingface)"
  type        = string
  default     = "groq"
  
  validation {
    condition     = contains(["groq", "openai", "huggingface"], var.llm_provider)
    error_message = "LLM provider must be one of: groq, openai, huggingface."
  }
}

variable "log_retention_days" {
  description = "Number of days to retain logs"
  type        = number
  default     = 30
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 30
}

variable "lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 256
}

variable "ecs_cpu" {
  description = "ECS task CPU units"
  type        = number
  default     = 256
}

variable "ecs_memory" {
  description = "ECS task memory in MB"
  type        = number
  default     = 512
}
