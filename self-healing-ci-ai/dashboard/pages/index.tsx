import { useState, useEffect } from 'react'

interface LogEntry {
  key: string
  lastModified: string
  size: number
  runId: string
}

interface AnalysisResult {
  run_id: string
  timestamp: string
  success: boolean
  analysis?: {
    root_cause: string
    explanation: string
    fix_steps: string[]
  }
  github_action?: {
    type: string
    url: string
    number: number
  }
}

export default function Dashboard() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      // In a real implementation, these would call AWS APIs via signed URLs
      // For demo purposes, showing mock data
      const mockLogs: LogEntry[] = [
        {
          key: 'logs/123456789.txt',
          lastModified: new Date().toISOString(),
          size: 2048,
          runId: '123456789'
        },
        {
          key: 'logs/123456788.txt',
          lastModified: new Date(Date.now() - 3600000).toISOString(),
          size: 1856,
          runId: '123456788'
        }
      ]

      const mockAnalyses: AnalysisResult[] = [
        {
          run_id: '123456789',
          timestamp: new Date().toISOString(),
          success: true,
          analysis: {
            root_cause: 'Test failure flag (FAIL_TEST=1) is set',
            explanation: 'The FAIL_TEST environment variable is set to 1, causing intentional test failure for demonstration purposes.',
            fix_steps: [
              'Set repository variable FAIL_TEST=0 in GitHub repository settings',
              'Or remove the FAIL_TEST environment variable completely',
              'Navigate to Settings > Secrets and variables > Actions > Variables tab'
            ]
          },
          github_action: {
            type: 'issue',
            url: 'https://github.com/user/repo/issues/1',
            number: 1
          }
        }
      ]

      setLogs(mockLogs)
      setAnalyses(mockAnalyses)
    } catch (err) {
      setError('Failed to fetch data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 text-xl mb-2">⚠️ Error</div>
          <p className="text-gray-600">{error}</p>
          <button 
            onClick={fetchData}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-6">
            <h1 className="text-3xl font-bold text-gray-900">
              🤖 Self-Healing CI/CD Dashboard
            </h1>
            <p className="mt-2 text-gray-600">
              Monitor AI-powered failure analysis and automated fixes
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="text-2xl">📊</div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Runs</p>
                <p className="text-2xl font-semibold text-gray-900">{logs.length}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="text-2xl">🤖</div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">AI Analyses</p>
                <p className="text-2xl font-semibold text-gray-900">{analyses.length}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="text-2xl">✅</div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Success Rate</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analyses.length > 0 ? Math.round(analyses.filter(a => a.success).length / analyses.length * 100) : 0}%
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="text-2xl">🔧</div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Auto Fixes</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {analyses.filter(a => a.github_action).length}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Analyses */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Recent AI Analyses</h2>
          </div>
          <div className="divide-y divide-gray-200">
            {analyses.length === 0 ? (
              <div className="px-6 py-12 text-center">
                <div className="text-gray-400 text-4xl mb-4">🤖</div>
                <p className="text-gray-500">No AI analyses yet</p>
                <p className="text-sm text-gray-400 mt-2">
                  Trigger a build failure to see AI analysis in action
                </p>
              </div>
            ) : (
              analyses.map((analysis) => (
                <div key={analysis.run_id} className="px-6 py-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          analysis.success 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {analysis.success ? '✅ Success' : '❌ Failed'}
                        </span>
                        <span className="text-sm text-gray-500">
                          Run #{analysis.run_id}
                        </span>
                        <span className="text-sm text-gray-500">
                          {new Date(analysis.timestamp).toLocaleString()}
                        </span>
                      </div>
                      
                      {analysis.analysis && (
                        <div className="mt-3">
                          <h4 className="text-sm font-medium text-gray-900">
                            🔍 Root Cause
                          </h4>
                          <p className="text-sm text-gray-600 mt-1">
                            {analysis.analysis.root_cause}
                          </p>
                          
                          <h4 className="text-sm font-medium text-gray-900 mt-3">
                            📖 Explanation
                          </h4>
                          <p className="text-sm text-gray-600 mt-1">
                            {analysis.analysis.explanation}
                          </p>
                          
                          <h4 className="text-sm font-medium text-gray-900 mt-3">
                            🔧 Fix Steps
                          </h4>
                          <ul className="text-sm text-gray-600 mt-1 space-y-1">
                            {analysis.analysis.fix_steps.map((step, index) => (
                              <li key={index} className="flex items-start">
                                <span className="text-gray-400 mr-2">•</span>
                                {step}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {analysis.github_action && (
                        <div className="mt-3">
                          <a
                            href={analysis.github_action.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800"
                          >
                            🔗 View {analysis.github_action.type} #{analysis.github_action.number}
                            <svg className="ml-1 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                          </a>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Recent Logs */}
        <div className="bg-white rounded-lg shadow mt-8">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Recent Log Files</h2>
          </div>
          <div className="divide-y divide-gray-200">
            {logs.length === 0 ? (
              <div className="px-6 py-12 text-center">
                <div className="text-gray-400 text-4xl mb-4">📄</div>
                <p className="text-gray-500">No log files yet</p>
                <p className="text-sm text-gray-400 mt-2">
                  Build logs will appear here after CI/CD runs
                </p>
              </div>
            ) : (
              logs.map((log) => (
                <div key={log.key} className="px-6 py-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        📄 {log.key}
                      </p>
                      <p className="text-sm text-gray-500">
                        Run #{log.runId} • {(log.size / 1024).toFixed(1)} KB • {new Date(log.lastModified).toLocaleString()}
                      </p>
                    </div>
                    <button className="text-sm text-blue-600 hover:text-blue-800">
                      View Log
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>
            💡 This is a demo dashboard. In production, integrate with AWS APIs for real-time data.
          </p>
        </div>
      </div>
    </div>
  )
}
