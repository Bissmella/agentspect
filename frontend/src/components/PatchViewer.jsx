const OP_COLORS = {
  add: 'text-green-700 bg-green-50',
  remove: 'text-red-700 bg-red-50',
  replace: 'text-amber-700 bg-amber-50',
  move: 'text-blue-700 bg-blue-50',
  copy: 'text-blue-700 bg-blue-50',
  test: 'text-gray-700 bg-gray-50',
};

export default function PatchViewer({ patchOps }) {
  if (!patchOps || patchOps.length === 0) {
    return <p className="text-sm text-gray-400 italic">No patches applied.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-1.5 px-2 font-medium text-gray-600">Op</th>
            <th className="text-left py-1.5 px-2 font-medium text-gray-600">Path</th>
            <th className="text-left py-1.5 px-2 font-medium text-gray-600">Value</th>
          </tr>
        </thead>
        <tbody>
          {patchOps.map((op, i) => (
            <tr key={i} className="border-b border-gray-100">
              <td className="py-1.5 px-2">
                <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-mono font-medium ${OP_COLORS[op.op] || ''}`}>
                  {op.op}
                </span>
              </td>
              <td className="py-1.5 px-2 font-mono text-gray-700">{op.path}</td>
              <td className="py-1.5 px-2 font-mono text-gray-600">
                {op.value !== undefined ? JSON.stringify(op.value) : '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
