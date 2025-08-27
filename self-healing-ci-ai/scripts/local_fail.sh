#!/bin/bash

# Script to trigger a failing CI/CD build for testing the self-healing system

set -e

echo "🧪 Triggering a failing CI/CD build for self-healing demo..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Warning: You're on branch '$CURRENT_BRANCH', not 'main'"
    echo "   The CI workflow is configured to run on main branch"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create a trigger comment to force a rebuild
echo "// Trigger rebuild - $(date)" >> app/src/server.js

# Export the failure flag
export FAIL_TEST=1

# Commit and push the change
git add .
git commit -m "test: trigger self-healing demo (FAIL_TEST=1)"

echo "📤 Pushing commit to trigger CI/CD pipeline..."
git push origin "$CURRENT_BRANCH"

echo ""
echo "✅ Failure trigger deployed!"
echo ""
echo "📋 What happens next:"
echo "   1. GitHub Actions will run the CI/CD pipeline"
echo "   2. Tests will fail due to FAIL_TEST=1"
echo "   3. Logs will be uploaded to S3"
echo "   4. AI analyzer Lambda will be invoked"
echo "   5. AI will analyze the logs and create a PR/Issue with fixes"
echo "   6. You'll receive an email with the analysis"
echo ""
echo "🔗 Monitor the pipeline at:"
echo "   https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/actions"
echo ""
echo "📧 Check your email (${SES_TO_EMAIL:-configured address}) for AI analysis results"
echo ""
echo "🔄 To fix and trigger a success build:"
echo "   1. Set repository variable FAIL_TEST=0 in GitHub settings"
echo "   2. Or run: git revert HEAD && git push"
