# SES email identity for the sender
resource "aws_ses_email_identity" "sender" {
  email = var.ses_from_email

  tags = local.common_tags
}

# SES email identity for the recipient (optional, for testing)
resource "aws_ses_email_identity" "recipient" {
  email = var.ses_to_email

  tags = local.common_tags
}

# SES configuration set for tracking
resource "aws_ses_configuration_set" "main" {
  name = "${var.project_prefix}-config-set"

  delivery_options {
    tls_policy = "Require"
  }

  tags = local.common_tags
}

# SES event destination for CloudWatch
resource "aws_ses_event_destination" "cloudwatch" {
  name                   = "cloudwatch-destination"
  configuration_set_name = aws_ses_configuration_set.main.name
  enabled                = true
  matching_types         = ["send", "reject", "bounce", "complaint", "delivery"]

  cloudwatch_destination {
    default_value  = "default"
    dimension_name = "MessageTag"
    value_source   = "messageTag"
  }
}

# Output SES verification instructions
output "ses_verification_instructions" {
  description = "Instructions for verifying SES email addresses"
  value = <<-EOT
    
    📧 SES EMAIL VERIFICATION REQUIRED:
    
    Before the system can send emails, you must verify your email addresses in SES:
    
    1. Go to the AWS SES Console:
       https://console.aws.amazon.com/ses/home?region=${var.aws_region}#verified-senders-email:
    
    2. Verify the sender email: ${var.ses_from_email}
       - Click "Create identity"
       - Select "Email address"
       - Enter: ${var.ses_from_email}
       - Click "Create identity"
       - Check your email and click the verification link
    
    3. Verify the recipient email: ${var.ses_to_email}
       - Repeat the same process for: ${var.ses_to_email}
    
    4. Note: If you're in the SES sandbox (new accounts), you can only send
       emails to verified addresses. To send to any address, request production access:
       https://console.aws.amazon.com/ses/home?region=${var.aws_region}#account-details:
    
    ⚠️  The AI analyzer will fail to send emails until verification is complete!
    
  EOT
}
