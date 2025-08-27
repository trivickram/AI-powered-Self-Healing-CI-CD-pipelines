import requests
import json
import logging
import base64
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GitHubClient:
    """
    Client for interacting with GitHub API to create PRs and Issues.
    """
    
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo  # format: "username/repository"
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🐙 GitHub Client initialized for repo: {repo}")
    
    def create_fix_pr_or_issue(self, analysis: Dict[str, Any], run_id: str, 
                              branch: str = 'main', commit_sha: str = '') -> Dict[str, Any]:
        """
        Create a Pull Request with fix or fallback to Issue creation.
        
        Args:
            analysis: AI analysis result containing fix suggestions
            run_id: GitHub Actions run ID
            branch: Source branch name
            commit_sha: Commit SHA that triggered the failure
            
        Returns:
            Dictionary with creation result and URL
        """
        try:
            # Try to create a PR first if we have a usable patch
            if analysis.get('optional_patch') and self._is_patch_safe(analysis['optional_patch']):
                logger.info("🔧 Attempting to create Pull Request with fix")
                return self._create_fix_pull_request(analysis, run_id, branch, commit_sha)
            else:
                logger.info("📝 Creating Issue with fix suggestions")
                return self._create_fix_issue(analysis, run_id, branch, commit_sha)
                
        except Exception as e:
            logger.error(f"❌ Error creating PR/Issue: {e}")
            # Fallback to basic issue creation
            return self._create_basic_issue(analysis, run_id, str(e))
    
    def _create_fix_pull_request(self, analysis: Dict[str, Any], run_id: str, 
                                branch: str, commit_sha: str) -> Dict[str, Any]:
        """
        Create a Pull Request with the suggested fix.
        """
        try:
            # Create a new branch for the fix
            fix_branch = f"ai/self-heal/{run_id}"
            
            # Get the reference of the base branch
            ref_response = requests.get(
                f"{self.base_url}/repos/{self.repo}/git/refs/heads/{branch}",
                headers=self.headers
            )
            ref_response.raise_for_status()
            base_sha = ref_response.json()['object']['sha']
            
            # Create new branch
            branch_data = {
                "ref": f"refs/heads/{fix_branch}",
                "sha": base_sha
            }
            
            branch_response = requests.post(
                f"{self.base_url}/repos/{self.repo}/git/refs",
                headers=self.headers,
                json=branch_data
            )
            branch_response.raise_for_status()
            
            # Apply the patch/fix (simplified - in reality would need file parsing)
            # For now, we'll create a simple fix file
            fix_content = self._generate_fix_file_content(analysis)
            
            # Create/update file with fix
            file_path = f"ai_fixes/fix_{run_id}.md"
            file_data = {
                "message": f"🤖 AI-generated fix for run {run_id}",
                "content": base64.b64encode(fix_content.encode()).decode(),
                "branch": fix_branch
            }
            
            file_response = requests.put(
                f"{self.base_url}/repos/{self.repo}/contents/{file_path}",
                headers=self.headers,
                json=file_data
            )
            file_response.raise_for_status()
            
            # Create Pull Request
            pr_data = {
                "title": f"🤖 AI Fix: {analysis.get('root_cause', 'CI/CD Failure')}",
                "head": fix_branch,
                "base": branch,
                "body": self._generate_pr_body(analysis, run_id),
                "draft": False
            }
            
            pr_response = requests.post(
                f"{self.base_url}/repos/{self.repo}/pulls",
                headers=self.headers,
                json=pr_data
            )
            pr_response.raise_for_status()
            
            pr_data = pr_response.json()
            
            logger.info(f"✅ Pull Request created: {pr_data['html_url']}")
            
            return {
                'type': 'pull_request',
                'number': pr_data['number'],
                'url': pr_data['html_url'],
                'branch': fix_branch,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create PR: {e}")
            # Fallback to issue creation
            return self._create_fix_issue(analysis, run_id, branch, commit_sha)
    
    def _create_fix_issue(self, analysis: Dict[str, Any], run_id: str, 
                         branch: str, commit_sha: str) -> Dict[str, Any]:
        """
        Create an Issue with fix suggestions.
        """
        try:
            issue_data = {
                "title": f"🤖 AI Analysis: {analysis.get('root_cause', 'CI/CD Failure')}",
                "body": self._generate_issue_body(analysis, run_id, branch, commit_sha),
                "labels": ["ai-analysis", "ci-cd-failure", "auto-generated"]
            }
            
            response = requests.post(
                f"{self.base_url}/repos/{self.repo}/issues",
                headers=self.headers,
                json=issue_data
            )
            response.raise_for_status()
            
            issue_data = response.json()
            
            logger.info(f"✅ Issue created: {issue_data['html_url']}")
            
            return {
                'type': 'issue',
                'number': issue_data['number'],
                'url': issue_data['html_url'],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create issue: {e}")
            return {'type': 'error', 'error': str(e), 'success': False}
    
    def _create_basic_issue(self, analysis: Dict[str, Any], run_id: str, error: str) -> Dict[str, Any]:
        """
        Create a basic issue when other methods fail.
        """
        try:
            issue_data = {
                "title": f"🚨 CI/CD Failure - Run {run_id}",
                "body": f"""## 🤖 AI Analysis Failed

**Run ID:** {run_id}
**Error:** {error}

**Root Cause:** {analysis.get('root_cause', 'Could not determine')}

**Suggested Actions:**
{chr(10).join(analysis.get('exact_fix', ['- Manual investigation required']))}

---
*This issue was automatically created by the self-healing CI/CD system.*
""",
                "labels": ["ci-cd-failure", "needs-investigation"]
            }
            
            response = requests.post(
                f"{self.base_url}/repos/{self.repo}/issues",
                headers=self.headers,
                json=issue_data
            )
            response.raise_for_status()
            
            issue_data = response.json()
            
            return {
                'type': 'basic_issue',
                'number': issue_data['number'],
                'url': issue_data['html_url'],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create basic issue: {e}")
            return {'type': 'error', 'error': str(e), 'success': False}
    
    def _is_patch_safe(self, patch: str) -> bool:
        """
        Check if the patch is safe to apply automatically.
        """
        if not patch or len(patch) > 2000:  # Too large
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'rm -rf', 'sudo', 'chmod 777', 'DELETE FROM', 'DROP TABLE',
            'eval(', 'exec(', 'system(', 'shell_exec('
        ]
        
        patch_lower = patch.lower()
        for pattern in dangerous_patterns:
            if pattern in patch_lower:
                return False
        
        return True
    
    def _generate_fix_file_content(self, analysis: Dict[str, Any]) -> str:
        """
        Generate content for the fix file.
        """
        return f"""# 🤖 AI-Generated Fix

## Root Cause
{analysis.get('root_cause', 'Unknown')}

## Explanation
{analysis.get('explanation', 'No explanation provided')}

## Exact Fix Steps
{chr(10).join(analysis.get('exact_fix', ['No fix steps provided']))}

## Optional Patch
```
{analysis.get('optional_patch', 'No patch provided')}
```

---
*Generated by AI Self-Healing CI/CD System*
"""
    
    def _generate_pr_body(self, analysis: Dict[str, Any], run_id: str) -> str:
        """
        Generate Pull Request body with fix details.
        """
        return f"""## 🤖 AI-Generated Fix for CI/CD Failure

**Run ID:** {run_id}

### 🔍 Root Cause
{analysis.get('root_cause', 'Unknown')}

### 📖 Explanation
{analysis.get('explanation', 'No explanation provided')}

### 🔧 Applied Fixes
{chr(10).join(analysis.get('exact_fix', ['No fix steps provided']))}

### 📝 Changes Made
This PR includes an AI-generated fix file with detailed instructions and optional code patches.

### ⚠️ Review Required
Please review this AI-generated fix carefully before merging. While the AI has analyzed the failure logs, human verification is recommended for all automated fixes.

---
*This Pull Request was automatically created by the self-healing CI/CD system.*
"""
    
    def _generate_issue_body(self, analysis: Dict[str, Any], run_id: str, 
                           branch: str, commit_sha: str) -> str:
        """
        Generate Issue body with analysis details.
        """
        return f"""## 🤖 AI Analysis for CI/CD Failure

**Run ID:** {run_id}
**Branch:** {branch}
**Commit:** {commit_sha[:8]}

### 🔍 Root Cause
{analysis.get('root_cause', 'Unknown')}

### 📖 Explanation
{analysis.get('explanation', 'No explanation provided')}

### 🔧 Suggested Fixes
{chr(10).join(analysis.get('exact_fix', ['No fix steps provided']))}

### 📄 Optional Patch
{f'```{chr(10)}{analysis.get("optional_patch", "No patch provided")}{chr(10)}```' if analysis.get('optional_patch') else 'No patch provided'}

### 🔗 Related Links
- [Failed Run](https://github.com/{self.repo}/actions/runs/{run_id})
- [Commit Details](https://github.com/{self.repo}/commit/{commit_sha})

---
*This issue was automatically created by the self-healing CI/CD system. Please review and apply the suggested fixes manually.*
"""
