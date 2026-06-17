import { useParams, Link } from 'react-router-dom';
import { useRun, useReport, useScenarios } from '../hooks/useRunQueries';
import VerdictSummary from '../components/VerdictSummary';
import MetricsDashboard from '../components/MetricsDashboard';
import ScenarioCard from '../components/ScenarioCard';
import ErrorAlert from '../components/ErrorAlert';

export default function Report() {
  const { id } = useParams();
  const { data: suite, isLoading: suiteLoading } = useRun(id);
  const { data: report, isLoading: reportLoading, error: reportError } = useReport(id);
  const { data: scenarios } = useScenarios(id);

  if (suiteLoading || reportLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <svg className="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
    );
  }

  if (reportError) {
    const msg = reportError.detail || 'Report not available yet.';
    return (
      <div>
        <Link to={`/runs/${id}`} className="text-sm text-blue-600 hover:text-blue-800 mb-4 inline-block">
          &larr; Back to Dashboard
        </Link>
        <ErrorAlert message={msg} />
      </div>
    );
  }

  const summary = report?.summary || {};
  const verdictCounts = summary.verdict_counts || {};
  const failureAnalysis = summary.failure_analysis;
  const probeChains = summary.probe_chains || [];

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <Link to={`/runs/${id}`} className="text-sm text-blue-600 hover:text-blue-800 mb-2 inline-block">
          &larr; Back to Dashboard
        </Link>
        <h1 className="text-2xl font-bold text-gray-900">
          Test Report {suite ? `— ${suite.agent_name}` : ''}
        </h1>
        {suite?.completed_at && (
          <p className="text-sm text-gray-500 mt-1">
            Completed {new Date(suite.completed_at).toLocaleString()}
          </p>
        )}
      </div>

      {/* Verdict summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 mb-6">
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Verdict Summary</h2>
        <VerdictSummary verdictCounts={verdictCounts} />
      </div>

      {/* Quantitative metrics */}
      {summary.metrics && (
        <div className="bg-white border border-gray-200 rounded-lg p-5 mb-6">
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Quantitative Metrics</h2>
          <MetricsDashboard metrics={summary.metrics} />
        </div>
      )}

      {/* Failure analysis */}
      {failureAnalysis && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-5 mb-6">
          <h2 className="text-sm font-semibold text-red-800 uppercase tracking-wide mb-2">Failure Analysis</h2>
          {typeof failureAnalysis === 'string' ? (
            <p className="text-sm text-red-700 whitespace-pre-wrap">{failureAnalysis}</p>
          ) : (
            <div className="text-sm text-red-700 space-y-2">
              {failureAnalysis.summary && <p>{failureAnalysis.summary}</p>}
              {failureAnalysis.details && (
                <ul className="list-disc list-inside space-y-1">
                  {(Array.isArray(failureAnalysis.details) ? failureAnalysis.details : []).map((d, i) => (
                    <li key={i}>{typeof d === 'string' ? d : JSON.stringify(d)}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}

      {/* Probe chains */}
      {probeChains.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-5 mb-6">
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Probe Chains</h2>
          <div className="space-y-2">
            {probeChains.map((chain, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="font-mono text-xs text-gray-500">{chain.primary_id || chain.primary}</span>
                <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
                <span className="font-mono text-xs text-gray-500">{chain.probe_id || chain.probe}</span>
                <span className="text-gray-600">= {chain.outcome || chain.verdict || 'pending'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-scenario breakdown */}
      <div>
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
          Scenarios ({scenarios?.length || 0})
        </h2>
        <div className="space-y-2">
          {(scenarios || []).map((scenario) => (
            <ScenarioCard key={scenario.id} scenario={scenario} runId={id} expandable />
          ))}
        </div>
      </div>
    </div>
  );
}
