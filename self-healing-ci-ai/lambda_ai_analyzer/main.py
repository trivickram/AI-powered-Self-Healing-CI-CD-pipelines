import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from s3_client import S3Client
from llm_client import LLMClient
from github_client import GitHubClient
from email_notifier import EmailNotifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for AI-powered CI/CD failure analysis.
    
    Expected event format:
    {
        "run_id": "123456789",
        "repo": "username/repository", 
        "branch": "main",
        "commit_sha": "abc123...",
        "trigger": "push"
    }
    """
    
    try:
        logger.info(f"🤖 AI Analyzer started - Event: {json.dumps(event, indent=2)}")
        
        # Extract event data
        run_id = event.get('run_id')
        repo = event.get('repo')
        branch = event.get('branch', 'main')
        commit_sha = event.get('commit_sha', '')
        trigger = event.get('trigger', 'unknown')
        
        if not run_id or not repo:
            raise ValueError("Missing required fields: run_id and repo")
        
        # Initialize clients
        s3_client = S3Client(
            bucket_name=os.environ['LOG_BUCKET'],
            region=os.environ['AWS_REGION']
        )
        
        llm_client = LLMClient(
            provider=os.environ.get('PROVIDER', 'groq'),
            api_key=os.environ.get('LLM_API_KEY')
        )
        
        github_client = GitHubClient(
            token=os.environ.get('GITHUB_TOKEN'),
            repo=repo
        )
        
        email_notifier = EmailNotifier(
            region=os.environ['AWS_REGION'],
            from_email=os.environ['SES_FROM_EMAIL'],
            to_email=os.environ['SES_TO_EMAIL']
        )
        
        # Step 1: Retrieve logs from S3
        logger.info(f"📋 Retrieving logs for run {run_id}")
        log_key = f"logs/{run_id}.txt"
        log_content = s3_client.get_log_file(log_key)
        
        if not log_content:
            raise Exception(f"No log file found for run {run_id}")
        
        logger.info(f"📄 Retrieved log file ({len(log_content)} characters)")
        
        # Step 2: Analyze with AI
        logger.info("🧠 Starting AI analysis...")
        analysis = llm_client.analyze_failure(
            log_content=log_content,
            repo=repo,
            branch=branch,
            run_id=run_id
        )
        
        if not analysis:
            raise Exception("AI analysis failed to return results")
        
        logger.info("✅ AI analysis completed")
        logger.info(f"🔍 Root cause: {analysis.get('root_cause', 'Unknown')}")
        
        # Step 3: Create GitHub PR or Issue
        logger.info("📝 Creating GitHub PR/Issue...")
        pr_issue_result = github_client.create_fix_pr_or_issue(
            analysis=analysis,
            run_id=run_id,
            branch=branch,
            commit_sha=commit_sha
        )
        
        # Step 4: Send email notification
        logger.info("📧 Sending email notification...")
        email_result = email_notifier.send_analysis_email(
            analysis=analysis,
            repo=repo,
            run_id=run_id,
            pr_issue_url=pr_issue_result.get('url', '')
        )
        
        # Step 5: Return results
        result = {
            'success': True,
            'run_id': run_id,
            'repo': repo,
            'analysis': {
                'root_cause': analysis.get('root_cause'),
                'explanation': analysis.get('explanation'),
                'fix_steps': analysis.get('exact_fix', [])
            },
            'github_action': pr_issue_result,
            'email_sent': email_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info("🎉 AI analysis pipeline completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"❌ AI Analyzer failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Try to send error notification
        try:
            if 'email_notifier' in locals():
                email_notifier.send_error_email(
                    error=str(e),
                    repo=event.get('repo', 'unknown'),
                    run_id=event.get('run_id', 'unknown')
                )
        except Exception as email_error:
            logger.error(f"Failed to send error email: {email_error}")
        
        return {
            'success': False,
            'error': str(e),
            'run_id': event.get('run_id'),
            'repo': event.get('repo'),
            'timestamp': datetime.utcnow().isoformat()
        }

def health_check(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Simple health check endpoint for the Lambda function.
    """
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'healthy',
            'function': 'self-healing-ai-analyzer',
            'timestamp': datetime.utcnow().isoformat(),
            'environment': {
                'provider': os.environ.get('PROVIDER', 'not_set'),
                'region': os.environ.get('AWS_REGION', 'not_set'),
                'bucket': os.environ.get('LOG_BUCKET', 'not_set')
            }
        })
    }
