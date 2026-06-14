import VerdictBadge from './VerdictBadge';

const VERDICT_BAR_COLORS = {
  success: 'bg-green-500',
  success_unverified: 'bg-emerald-400',
  failure: 'bg-red-500',
  failure_corrupt: 'bg-red-700',
  suspect: 'bg-amber-500',
  error: 'bg-gray-400',
};

const VERDICT_ORDER = ['success', 'success_unverified', 'failure', 'failure_corrupt', 'suspect', 'error'];

export default function VerdictSummary({ verdictCounts }) {
  if (!verdictCounts) return null;

  const total = Object.values(verdictCounts).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  return (
    <div>
      <div className="flex flex-wrap gap-3 mb-3">
        {VERDICT_ORDER.map((v) =>
          verdictCounts[v] ? (
            <div key={v} className="flex items-center gap-1.5">
              <VerdictBadge verdict={v} />
              <span className="text-sm font-semibold text-gray-700">{verdictCounts[v]}</span>
            </div>
          ) : null
        )}
        <div className="flex items-center gap-1.5 ml-auto">
          <span className="text-sm text-gray-500">Total:</span>
          <span className="text-sm font-semibold text-gray-700">{total}</span>
        </div>
      </div>
      <div className="w-full h-3 rounded-full bg-gray-200 flex overflow-hidden">
        {VERDICT_ORDER.map((v) => {
          const count = verdictCounts[v] || 0;
          if (count === 0) return null;
          const pct = (count / total) * 100;
          return (
            <div
              key={v}
              className={`${VERDICT_BAR_COLORS[v]} transition-all duration-500`}
              style={{ width: `${pct}%` }}
              title={`${v}: ${count}`}
            />
          );
        })}
      </div>
    </div>
  );
}
