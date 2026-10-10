import { useParams, Link } from 'react-router-dom';
import { useRun } from '../hooks/useRunQueries';
import { useRunWebSocket } from '../hooks/useRunWebSocket';
import StatusBadge from '../components/StatusBadge';
import ProgressBar from '../components/ProgressBar';
import EventTimeline from '../components/EventTimeline';
import ScenarioCard from '../components/ScenarioCard';
import VerdictSummary from '../components/VerdictSummary';
import ErrorAlert from '../components/ErrorAlert';
import Spinner from '../components/Spinner';

export default function RunDashboard() {
  const { id } = useParams();
  const { events, connectionStatus, isTerminal, latestEvent } = useRunWebSocket(id);
  const { data: suite, isLoading } = useRun(id, {
    poll: !isTerminal && connectionStatus === 'disconnected',
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner className="h-8 w-8 text-brand-600" />
      </div>
    );
  }

  if (!suite) {
    return <ErrorAlert message="Suite not found." />;
  }

  const completedCount = (suite.scenarios || []).filter(
    (s) => s.status === 'completed' || s.status === 'skipped'
  ).length;

  const completedEvent = latestEvent?.event === 'suite_completed' ? latestEvent : null;
  const failedEvent = latestEvent?.event === 'suite_failed' ? latestEvent : null;

  return (
    <div>
      {/* Suite header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{suite.agent_name}</h1>
            <div className="flex items-center gap-3 mt-1 text-sm text-gray-500">
              <span className="font-mono">{suite.protocol}</span>
              <span>{suite.llm_provider} / {suite.llm_model}</span>
              <StatusBadge status={suite.status} />
            </div>
          </div>
          {(suite.status === 'completed' || completedEvent) && (
            <Link to={`/runs/${id}/report`} className="btn-success">
              View Report
            </Link>
          )}
        </div>
      </div>

      {/* Connection banner */}
      {connectionStatus === 'disconnected' && !isTerminal && (
        <div className="mb-4 rounded-md bg-yellow-50 border border-yellow-200 p-3">
          <p className="text-sm text-yellow-800">Connection lost. Reconnecting...</p>
        </div>
      )}

      {/* Terminal banners */}
      {completedEvent && (
        <div className="mb-6 rounded-lg bg-green-50 border border-green-200 p-4">
          <h3 className="text-sm font-semibold text-green-800 mb-2">Suite Completed</h3>
          <VerdictSummary verdictCounts={completedEvent.data?.verdict_counts} />
        </div>
      )}

      {failedEvent && (
        <ErrorAlert message={`Suite failed: ${failedEvent.data?.error || 'Unknown error'}`} />
      )}

      {/* Progress */}
      <div className="mb-6">
        <ProgressBar completed={completedCount} total={suite.total_scenarios} />
        <div className="flex gap-4 mt-2 text-xs text-gray-500">
          <span>Positive: {suite.positive_count}</span>
          <span>Negative: {suite.negative_count}</span>
        </div>
      </div>

      {/* Main content: timeline + scenarios */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Event timeline */}
        <div className="lg:col-span-2">
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Events</h2>
          <EventTimeline events={events} />
        </div>

        {/* Scenario list */}
        <div className="lg:col-span-3">
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
            Scenarios ({suite.scenarios?.length || 0})
          </h2>
          <div className="space-y-2">
            {(suite.scenarios || []).map((scenario) => (
              <ScenarioCard key={scenario.id} scenario={scenario} runId={id} />
            ))}
            {(!suite.scenarios || suite.scenarios.length === 0) && (
              <p className="text-sm text-gray-400 italic">Waiting for scenarios to be generated...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
