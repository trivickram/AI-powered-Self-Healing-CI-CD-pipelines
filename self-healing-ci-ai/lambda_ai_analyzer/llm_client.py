import os
import json
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Provider-agnostic LLM client supporting multiple providers.
    """
    
    def __init__(self, provider: str = 'groq', api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.timeout = 30  # seconds
        
        # Load prompts
        self.system_prompt = self._load_prompt('system_prompt.txt')
        self.user_prompt_template = self._load_prompt('user_prompt_template.txt')
        
        logger.info(f"🤖 LLM Client initialized with provider: {self.provider}")
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', filename)
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt {filename}: {e}")
            return self._get_default_prompt(filename)
    
    def _get_default_prompt(self, filename: str) -> str:
        """Fallback prompts if files are not available."""
        if 'system' in filename:
            return ("You are a CI/CD failure analyst. You read raw logs from build/test/deploy jobs "
                   "and produce short, precise diagnoses and concrete fixes. Be specific. "
                   "Prefer minimal, safe changes. NEVER hallucinate file paths; infer from context.")
        else:
            return ("Context:\n- Repo: {{repo}}\n- Branch: {{branch}}\n- Run ID: {{run_id}}\n\n"
                   "Task:\nAnalyze the CI/CD log below and produce:\n"
                   "1) Root cause (one line)\n"
                   "2) Explanation (3–6 short lines)\n"
                   "3) Exact actionable fix (bullet list with precise file/line/YAML keys or commands)\n"
                   "4) (Optional) A minimal patch or code/YAML snippet to apply\n\n"
                   "Log:\n{{log_text}}")
    
    def analyze_failure(self, log_content: str, repo: str, branch: str, run_id: str) -> Dict[str, Any]:
        """
        Analyze CI/CD failure using the configured LLM provider.
        """
        try:
            # Prepare the prompt
            user_prompt = self.user_prompt_template.format(
                repo=repo,
                branch=branch,
                run_id=run_id,
                log_text=log_content[-4000:]  # Limit log size to avoid token limits
            )
            
            logger.info(f"🔍 Analyzing failure with {self.provider}")
            
            # Call the appropriate provider
            if self.provider == 'groq':
                response = self._call_groq(user_prompt)
            elif self.provider == 'openai':
                response = self._call_openai(user_prompt)
            elif self.provider == 'huggingface':
                response = self._call_huggingface(user_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Parse the response
            analysis = self._parse_analysis_response(response)
            
            logger.info("✅ LLM analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ LLM analysis failed: {e}")
            # Return a fallback analysis
            return self._generate_fallback_analysis(log_content)
    
    def _call_groq(self, user_prompt: str) -> str:
        """Call Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3-8b-8192",  # Fast and capable model
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_p": 1,
            "stop": None
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _call_huggingface(self, user_prompt: str) -> str:
        """Call HuggingFace Inference API."""
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Combine system and user prompts for HuggingFace
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        
        data = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.1,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        return str(result)
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured analysis.
        """
        lines = response.strip().split('\n')
        analysis = {
            'root_cause': '',
            'explanation': '',
            'exact_fix': [],
            'optional_patch': ''
        }
        
        current_section = None
        explanation_lines = []
        fix_lines = []
        patch_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'root cause' in line.lower() or line.startswith('1)'):
                current_section = 'root_cause'
                # Extract the root cause from the same line if present
                if ':' in line:
                    analysis['root_cause'] = line.split(':', 1)[1].strip()
                continue
            elif 'explanation' in line.lower() or line.startswith('2)'):
                current_section = 'explanation'
                continue
            elif 'fix' in line.lower() or line.startswith('3)'):
                current_section = 'fix'
                continue
            elif 'patch' in line.lower() or line.startswith('4)'):
                current_section = 'patch'
                continue
            
            # Assign content to sections
            if current_section == 'root_cause' and not analysis['root_cause']:
                analysis['root_cause'] = line
            elif current_section == 'explanation':
                explanation_lines.append(line)
            elif current_section == 'fix':
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    fix_lines.append(line)
                else:
                    fix_lines.append(f"- {line}")
            elif current_section == 'patch':
                patch_lines.append(line)
        
        # Combine multi-line sections
        analysis['explanation'] = '\n'.join(explanation_lines[:6])  # Limit to 6 lines
        analysis['exact_fix'] = fix_lines[:8]  # Limit to 8 fix steps
        analysis['optional_patch'] = '\n'.join(patch_lines)
        
        # Fallbacks for missing data
        if not analysis['root_cause']:
            analysis['root_cause'] = "CI/CD pipeline failure detected"
        
        if not analysis['explanation']:
            analysis['explanation'] = "The build or test process encountered an error. Check the logs for specific error messages."
        
        if not analysis['exact_fix']:
            analysis['exact_fix'] = [
                "- Review the error logs for specific failure points",
                "- Check dependencies and environment configuration",
                "- Verify all required secrets and environment variables are set"
            ]
        
        return analysis
    
    def _generate_fallback_analysis(self, log_content: str) -> Dict[str, Any]:
        """
        Generate a basic analysis when LLM fails.
        """
        logger.warning("🔄 Generating fallback analysis")
        
        # Simple keyword-based analysis
        log_lower = log_content.lower()
        
        if 'fail_test=1' in log_lower:
            return {
                'root_cause': "Test failure flag (FAIL_TEST=1) is set",
                'explanation': "The FAIL_TEST environment variable is set to 1, causing intentional test failure for demonstration purposes.",
                'exact_fix': [
                    "- Set repository variable FAIL_TEST=0 in GitHub repository settings",
                    "- Or remove the FAIL_TEST environment variable completely",
                    "- Navigate to Settings > Secrets and variables > Actions > Variables tab"
                ],
                'optional_patch': ""
            }
        elif 'npm test' in log_lower and 'exit code 1' in log_lower:
            return {
                'root_cause': "NPM test suite failed",
                'explanation': "One or more tests in the test suite are failing, causing the CI pipeline to exit with code 1.",
                'exact_fix': [
                    "- Run 'npm test' locally to identify failing tests",
                    "- Check test files in the 'tests/' directory",
                    "- Verify application is running correctly before tests execute",
                    "- Check for missing dependencies or environment variables"
                ],
                'optional_patch': ""
            }
        elif 'docker' in log_lower and ('error' in log_lower or 'failed' in log_lower):
            return {
                'root_cause': "Docker build or deployment failed",
                'explanation': "The Docker container build process or deployment to ECS encountered an error.",
                'exact_fix': [
                    "- Check Dockerfile syntax and base image availability",
                    "- Verify ECR repository exists and permissions are correct",
                    "- Check ECS service configuration and task definition",
                    "- Ensure AWS credentials have sufficient permissions"
                ],
                'optional_patch': ""
            }
        else:
            return {
                'root_cause': "CI/CD pipeline failure - generic error",
                'explanation': "The pipeline failed but specific error type could not be determined automatically. Manual investigation required.",
                'exact_fix': [
                    "- Review the complete build logs for error messages",
                    "- Check GitHub Actions workflow configuration",
                    "- Verify all required secrets and environment variables",
                    "- Test the build process locally to reproduce the issue"
                ],
                'optional_patch': ""
            }
