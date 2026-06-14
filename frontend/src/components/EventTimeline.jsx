import { useEffect, useRef } from 'react';
import { EVENT_LABELS } from '../constants';

function formatTime(timestamp) {
  const d = new Date(timestamp * 1000);
  return d.toLocaleTimeString();
}

function eventColor(eventName) {
  if (eventName === 'suite_completed') return 'bg-green-500';
  if (eventName === 'suite_failed' || eventName === 'patch_failed') return 'bg-red-500';
  return 'bg-blue-500';
}

export default function EventTimeline({ events }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  if (events.length === 0) {
    return <p className="text-sm text-gray-400 italic">Waiting for events...</p>;
  }

  return (
    <div className="max-h-[500px] overflow-y-auto pr-2">
      <div className="relative">
        <div className="absolute left-2.5 top-0 bottom-0 w-0.5 bg-gray-200" />
        <ul className="space-y-3">
          {events.map((ev, i) => (
            <li key={i} className="relative flex items-start pl-8">
              <div className={`absolute left-1 top-1.5 w-3 h-3 rounded-full ${eventColor(ev.event)} ring-2 ring-white`} />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-gray-900">
                  {EVENT_LABELS[ev.event] || ev.event}
                </p>
                <p className="text-xs text-gray-500">{formatTime(ev.timestamp)}</p>
                {ev.data?.error && (
                  <p className="text-xs text-red-600 mt-0.5">{ev.data.error}</p>
                )}
              </div>
            </li>
          ))}
          <div ref={endRef} />
        </ul>
      </div>
    </div>
  );
}
