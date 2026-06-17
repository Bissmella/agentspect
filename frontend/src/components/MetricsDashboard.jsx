function RateCard({ label, rate, numerator, denominator }) {
  const pct = Math.round(rate * 100);
  const color = pct >= 80 ? 'text-green-600' : pct >= 60 ? 'text-amber-600' : 'text-red-600';
  const bg = pct >= 80 ? 'bg-green-50' : pct >= 60 ? 'bg-amber-50' : 'bg-red-50';

  return (
    <div className={`${bg} rounded-lg p-4 flex flex-col items-center`}>
      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">{label}</span>
      <span className={`text-3xl font-bold ${color}`}>
        {denominator > 0 ? `${pct}%` : 'N/A'}
      </span>
      <span className="text-xs text-gray-400 mt-1">{numerator} / {denominator}</span>
    </div>
  );
}

function ConstraintTable({ violations }) {
  if (!violations || violations.length === 0) return null;

  return (
    <div className="mt-5">
      <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">
        Constraint Violation Breakdown
      </h3>
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 text-left text-xs text-gray-500 uppercase tracking-wide">
              <th className="px-3 py-2 font-medium">Boundary Tested</th>
              <th className="px-3 py-2 font-medium w-24 text-right">Violated</th>
              <th className="px-3 py-2 font-medium w-32">Rate</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {violations.map((v, i) => {
              const pct = Math.round(v.rate * 100);
              return (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-gray-700">{v.constraint}</td>
                  <td className="px-3 py-2 text-right text-gray-600">{v.violated} / {v.tested}</td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${pct >= 50 ? 'bg-red-500' : pct > 0 ? 'bg-amber-400' : 'bg-green-400'}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 w-8 text-right">{pct}%</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const RECOVERY_LABELS = {
  clean_refusal: { label: 'Clean Refusal', color: 'bg-green-500' },
  confused_response: { label: 'Confused', color: 'bg-amber-500' },
  error_response: { label: 'Error', color: 'bg-red-500' },
  information_leak: { label: 'Info Leak', color: 'bg-red-700' },
  unclassified: { label: 'Unclassified', color: 'bg-gray-400' },
};

function RecoveryBreakdown({ recovery }) {
  if (!recovery || recovery.total_adversarial === 0) return null;

  const segments = ['clean_refusal', 'confused_response', 'error_response', 'information_leak', 'unclassified']
    .map((key) => ({ key, count: recovery[key] || 0, ...RECOVERY_LABELS[key] }))
    .filter((s) => s.count > 0);

  return (
    <div className="mt-5">
      <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">
        Recovery Behavior
        <span className="ml-2 font-normal text-gray-400">
          ({Math.round(recovery.rate * 100)}% clean refusal rate)
        </span>
      </h3>
      <div className="w-full h-4 rounded-full bg-gray-200 flex overflow-hidden mb-2">
        {segments.map((s) => (
          <div
            key={s.key}
            className={`${s.color} transition-all duration-500`}
            style={{ width: `${(s.count / recovery.total_adversarial) * 100}%` }}
            title={`${s.label}: ${s.count}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-3">
        {segments.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5 text-xs text-gray-600">
            <span className={`w-2.5 h-2.5 rounded-full ${s.color}`} />
            {s.label}: {s.count}
          </div>
        ))}
      </div>
    </div>
  );
}

function EfficiencyCards({ efficiency }) {
  if (!efficiency) return null;
  const { avg_turns_to_completion, avg_turns_to_refusal } = efficiency;
  if (avg_turns_to_completion == null && avg_turns_to_refusal == null) return null;

  return (
    <div className="mt-5">
      <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">
        Conversation Efficiency
      </h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <span className="text-xs text-gray-500 block mb-1">Avg Turns to Completion</span>
          <span className="text-2xl font-bold text-blue-600">
            {avg_turns_to_completion != null ? avg_turns_to_completion.toFixed(1) : 'N/A'}
          </span>
          <span className="text-xs text-gray-400 block">positive scenarios</span>
        </div>
        <div className="bg-purple-50 rounded-lg p-3 text-center">
          <span className="text-xs text-gray-500 block mb-1">Avg Turns to Refusal</span>
          <span className="text-2xl font-bold text-purple-600">
            {avg_turns_to_refusal != null ? avg_turns_to_refusal.toFixed(1) : 'N/A'}
          </span>
          <span className="text-xs text-gray-400 block">negative scenarios</span>
        </div>
      </div>
    </div>
  );
}

export default function MetricsDashboard({ metrics }) {
  if (!metrics) return null;

  return (
    <div>
      <div className="grid grid-cols-3 gap-3">
        <RateCard
          label="Task Completion"
          rate={metrics.task_completion.rate}
          numerator={metrics.task_completion.successful}
          denominator={metrics.task_completion.total}
        />
        <RateCard
          label="Boundary Adherence"
          rate={metrics.boundary_adherence.rate}
          numerator={metrics.boundary_adherence.successful}
          denominator={metrics.boundary_adherence.total}
        />
        <RateCard
          label="Verification Rate"
          rate={metrics.verification_rate.rate}
          numerator={metrics.verification_rate.confirmed}
          denominator={metrics.verification_rate.total}
        />
      </div>
      <ConstraintTable violations={metrics.constraint_violations} />
      <RecoveryBreakdown recovery={metrics.recovery_behavior} />
      <EfficiencyCards efficiency={metrics.conversation_efficiency} />
    </div>
  );
}
