export default function TranscriptViewer({ turns }) {
  if (!turns || turns.length === 0) {
    return <p className="text-sm text-gray-400 italic">No conversation turns.</p>;
  }

  return (
    <div className="space-y-3 max-h-96 overflow-y-auto p-3 bg-gray-50 rounded-lg">
      {turns.map((turn, i) => (
        <div key={i} className="space-y-2">
          {turn.user && (
            <div className="flex justify-start">
              <div className="max-w-[75%] rounded-lg px-3 py-2 bg-blue-100 text-blue-900">
                <p className="text-xs font-medium text-blue-600 mb-0.5">User</p>
                <p className="text-sm whitespace-pre-wrap">{turn.user}</p>
              </div>
            </div>
          )}
          {turn.agent && (
            <div className="flex justify-end">
              <div className="max-w-[75%] rounded-lg px-3 py-2 bg-white border border-gray-200 text-gray-900">
                <p className="text-xs font-medium text-gray-500 mb-0.5">Agent</p>
                <p className="text-sm whitespace-pre-wrap">{turn.agent}</p>
                {turn.latency_ms > 0 && (
                  <p className="text-xs text-gray-400 mt-1">{turn.latency_ms}ms</p>
                )}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
