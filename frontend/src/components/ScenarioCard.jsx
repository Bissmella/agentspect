import { useState } from 'react';
import VerdictBadge from './VerdictBadge';
import StatusBadge from './StatusBadge';
import PatchViewer from './PatchViewer';
import TranscriptViewer from './TranscriptViewer';
import { useTranscript } from '../hooks/useRunQueries';

export default function ScenarioCard({ scenario, runId, expandable = false }) {
  const [expanded, setExpanded] = useState(false);
  const [showTranscript, setShowTranscript] = useState(false);

  const { data: transcript } = useTranscript(
    showTranscript ? runId : null,
    showTranscript ? scenario.id : null,
  );

  return (
    <div className="border border-gray-200 rounded-lg bg-white">
      <div
        className={`p-4 ${expandable ? 'cursor-pointer hover:bg-gray-50' : ''}`}
        onClick={() => expandable && setExpanded(!expanded)}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-gray-900 truncate">{scenario.description}</p>
            {scenario.verdict_reason && !expanded && (
              <p className="text-xs text-gray-500 mt-0.5 truncate">{scenario.verdict_reason}</p>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${scenario.type === 'positive' ? 'bg-blue-50 text-blue-700' : 'bg-orange-50 text-orange-700'}`}>
              {scenario.type}
            </span>
            <StatusBadge status={scenario.status} />
            <VerdictBadge verdict={scenario.verdict} />
            {expandable && (
              <svg className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-gray-200 p-4 space-y-4">
          {scenario.verdict_reason && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">Verdict Reason</h4>
              <p className="text-sm text-gray-700">{scenario.verdict_reason}</p>
            </div>
          )}

          {scenario.depends_on && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">Dependency</h4>
              <p className="text-sm text-gray-600">
                Depends on <span className="font-mono text-xs">{scenario.depends_on}</span>
                {scenario.depends_on_type && ` (${scenario.depends_on_type})`}
              </p>
            </div>
          )}

          {scenario.assertions_json && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">Assertions</h4>
              <ul className="space-y-1">
                {(Array.isArray(scenario.assertions_json) ? scenario.assertions_json : []).map((a, i) => (
                  <li key={i} className="text-sm text-gray-700 flex items-start gap-1.5">
                    <span className="font-mono text-xs px-1 py-0.5 rounded bg-gray-100 text-gray-600 flex-shrink-0">{a.type}</span>
                    <span>{a.description || a.check || a.expected_behavior || '-'}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {scenario.patch_ops_json && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">World State Patches</h4>
              <PatchViewer patchOps={scenario.patch_ops_json} />
            </div>
          )}

          <div>
            <button
              onClick={(e) => { e.stopPropagation(); setShowTranscript(!showTranscript); }}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              {showTranscript ? 'Hide Transcript' : 'View Transcript'}
            </button>
            {showTranscript && transcript && (
              <div className="mt-2">
                <TranscriptViewer turns={transcript.turns} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
