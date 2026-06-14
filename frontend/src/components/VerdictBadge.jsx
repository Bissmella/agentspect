import { VERDICT_COLORS } from '../constants';

export default function VerdictBadge({ verdict }) {
  if (!verdict) return null;

  const style = VERDICT_COLORS[verdict] || { bg: 'bg-gray-100', text: 'text-gray-600', label: verdict };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${style.bg} ${style.text}`}>
      {style.label}
    </span>
  );
}
