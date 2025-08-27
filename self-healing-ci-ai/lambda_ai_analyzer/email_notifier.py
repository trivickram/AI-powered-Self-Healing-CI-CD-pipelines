import boto3
import json
import logging
from typing import Dict, Any
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class EmailNotifier:
    """
    Client for sending email notifications via Amazon SES.
    """
    
    def __init__(self, region: str, from_email: str, to_email: str):
        self.region = region
        self.from_email = from_email
        self.to_email = to_email
        self.ses_client = boto3.client('ses', region_name=region)
        
        logger.info(f"📧 Email Notifier initialized - from: {from_email}, to: {to_email}")
    
    def send_analysis_email(self, analysis: Dict[str, Any], repo: str, 
                           run_id: str, pr_issue_url: str = '') -> bool:
        """
        Send email notification with AI analysis results.
        
        Args:
            analysis: AI analysis result dictionary
            repo: GitHub repository name
            run_id: GitHub Actions run ID
            pr_issue_url: URL to created PR or Issue
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            subject = f"🤖 [Self-Heal] {repo} - Run {run_id}"
            
            # Generate email body
            body_text = self._generate_text_body(analysis, repo, run_id, pr_issue_url)
            body_html = self._generate_html_body(analysis, repo, run_id, pr_issue_url)
            
            logger.info(f"📤 Sending analysis email for run {run_id}")
            
            response = self.ses_client.send_email(
                Source=self.from_email,
                Destination={'ToAddresses': [self.to_email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {
                        'Text': {'Data': body_text, 'Charset': 'UTF-8'},
                        'Html': {'Data': body_html, 'Charset': 'UTF-8'}
                    }
                }
            )
            
            message_id = response['MessageId']
            logger.info(f"✅ Email sent successfully - Message ID: {message_id}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'MessageRejected':
                logger.error("❌ Email rejected - check sender verification")
            elif error_code == 'MailFromDomainNotVerified':
                logger.error("❌ Sender domain not verified in SES")
            else:
                logger.error(f"❌ SES error: {e}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Unexpected error sending email: {e}")
            return False
    
    def send_error_email(self, error: str, repo: str, run_id: str) -> bool:
        """
        Send email notification about AI analyzer errors.
        
        Args:
            error: Error message
            repo: GitHub repository name
            run_id: GitHub Actions run ID
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            subject = f"🚨 [Self-Heal Error] {repo} - Run {run_id}"
            
            body_text = f"""
AI Self-Healing System Error

Repository: {repo}
Run ID: {run_id}
Error: {error}

The AI analysis system encountered an error while processing the CI/CD failure.
Please check the Lambda logs for more details and investigate the issue manually.

GitHub Actions Run: https://github.com/{repo}/actions/runs/{run_id}
"""
            
            body_html = f"""
<html>
<body>
<h2>🚨 AI Self-Healing System Error</h2>

<p><strong>Repository:</strong> {repo}</p>
<p><strong>Run ID:</strong> {run_id}</p>
<p><strong>Error:</strong> {error}</p>

<p>The AI analysis system encountered an error while processing the CI/CD failure.
Please check the Lambda logs for more details and investigate the issue manually.</p>

<p><a href="https://github.com/{repo}/actions/runs/{run_id}">View GitHub Actions Run</a></p>
</body>
</html>
"""
            
            logger.info(f"📤 Sending error notification for run {run_id}")
            
            response = self.ses_client.send_email(
                Source=self.from_email,
                Destination={'ToAddresses': [self.to_email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {
                        'Text': {'Data': body_text, 'Charset': 'UTF-8'},
                        'Html': {'Data': body_html, 'Charset': 'UTF-8'}
                    }
                }
            )
            
            message_id = response['MessageId']
            logger.info(f"✅ Error email sent successfully - Message ID: {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send error email: {e}")
            return False
    
    def send_success_email(self, repo: str, run_id: str, deployment_info: Dict[str, Any]) -> bool:
        """
        Send email notification for successful deployments.
        
        Args:
            repo: GitHub repository name
            run_id: GitHub Actions run ID
            deployment_info: Information about the deployment
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            subject = f"✅ [Deployment Success] {repo} - Run {run_id}"
            
            body_text = f"""
Deployment Successful!

Repository: {repo}
Run ID: {run_id}
Status: Successfully deployed to ECS Fargate

The application has been successfully built, tested, and deployed.

GitHub Actions Run: https://github.com/{repo}/actions/runs/{run_id}
"""
            
            body_html = f"""
<html>
<body>
<h2>✅ Deployment Successful!</h2>

<p><strong>Repository:</strong> {repo}</p>
<p><strong>Run ID:</strong> {run_id}</p>
<p><strong>Status:</strong> Successfully deployed to ECS Fargate</p>

<p>The application has been successfully built, tested, and deployed.</p>

<p><a href="https://github.com/{repo}/actions/runs/{run_id}">View GitHub Actions Run</a></p>
</body>
</html>
"""
            
            logger.info(f"📤 Sending success notification for run {run_id}")
            
            response = self.ses_client.send_email(
                Source=self.from_email,
                Destination={'ToAddresses': [self.to_email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': {
                        'Text': {'Data': body_text, 'Charset': 'UTF-8'},
                        'Html': {'Data': body_html, 'Charset': 'UTF-8'}
                    }
                }
            )
            
            message_id = response['MessageId']
            logger.info(f"✅ Success email sent successfully - Message ID: {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send success email: {e}")
            return False
    
    def _generate_text_body(self, analysis: Dict[str, Any], repo: str, 
                           run_id: str, pr_issue_url: str) -> str:
        """
        Generate plain text email body.
        """
        text_body = f"""
AI Self-Healing CI/CD Analysis

Repository: {repo}
Run ID: {run_id}

ROOT CAUSE:
{analysis.get('root_cause', 'Unknown')}

EXPLANATION:
{analysis.get('explanation', 'No explanation provided')}

SUGGESTED FIXES:
{chr(10).join(analysis.get('exact_fix', ['No fix steps provided']))}
"""
        
        if analysis.get('optional_patch'):
            text_body += f"""

OPTIONAL PATCH:
{analysis.get('optional_patch')}
"""
        
        if pr_issue_url:
            text_body += f"""

GITHUB ACTION:
{pr_issue_url}
"""
        
        text_body += f"""

GitHub Actions Run: https://github.com/{repo}/actions/runs/{run_id}

---
This analysis was automatically generated by the AI Self-Healing CI/CD system.
"""
        
        return text_body
    
    def _generate_html_body(self, analysis: Dict[str, Any], repo: str, 
                           run_id: str, pr_issue_url: str) -> str:
        """
        Generate HTML email body.
        """
        # Format fix steps as HTML list
        fix_steps_html = ""
        if analysis.get('exact_fix'):
            fix_steps_html = "<ul>"
            for step in analysis.get('exact_fix', []):
                step_clean = step.lstrip('- • * ')
                fix_steps_html += f"<li>{step_clean}</li>"
            fix_steps_html += "</ul>"
        else:
            fix_steps_html = "<p>No fix steps provided</p>"
        
        html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .code {{ background: #f8f8f8; padding: 10px; border-radius: 3px; font-family: monospace; }}
        .button {{ background: #007cba; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🤖 AI Self-Healing CI/CD Analysis</h2>
        <p><strong>Repository:</strong> {repo}</p>
        <p><strong>Run ID:</strong> {run_id}</p>
    </div>
    
    <div class="section">
        <h3>🔍 Root Cause</h3>
        <p>{analysis.get('root_cause', 'Unknown')}</p>
    </div>
    
    <div class="section">
        <h3>📖 Explanation</h3>
        <p>{analysis.get('explanation', 'No explanation provided').replace(chr(10), '<br>')}</p>
    </div>
    
    <div class="section">
        <h3>🔧 Suggested Fixes</h3>
        {fix_steps_html}
    </div>
"""
        
        if analysis.get('optional_patch'):
            html_body += f"""
    <div class="section">
        <h3>📄 Optional Patch</h3>
        <div class="code">{analysis.get('optional_patch').replace(chr(10), '<br>')}</div>
    </div>
"""
        
        if pr_issue_url:
            html_body += f"""
    <div class="section">
        <h3>🔗 GitHub Action</h3>
        <p><a href="{pr_issue_url}" class="button">View Pull Request/Issue</a></p>
    </div>
"""
        
        html_body += f"""
    <div class="section">
        <p><a href="https://github.com/{repo}/actions/runs/{run_id}" class="button">View GitHub Actions Run</a></p>
    </div>
    
    <hr>
    <p><em>This analysis was automatically generated by the AI Self-Healing CI/CD system.</em></p>
</body>
</html>
"""
        
        return html_body
