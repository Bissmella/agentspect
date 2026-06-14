import { STATUS_COLORS } from '../constants';

export default function StatusBadge({ status }) {
  if (!status) return null;

  const style = STATUS_COLORS[status] || { bg: 'bg-gray-100', text: 'text-gray-600', label: status };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${style.bg} ${style.text} ${style.animate ? 'animate-pulse' : ''}`}>
      {style.label}
    </span>
  );
}
