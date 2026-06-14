import { useState, useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';

const TERMINAL_EVENTS = new Set(['suite_completed', 'suite_failed']);
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];

export function useRunWebSocket(runId) {
  const [events, setEvents] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [isTerminal, setIsTerminal] = useState(false);

  const wsRef = useRef(null);
  const retryRef = useRef(0);
  const timerRef = useRef(null);
  const seenSeqs = useRef(new Set());
  const queryClient = useQueryClient();

  const connect = useCallback(() => {
    if (!runId || isTerminal) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/api/runs/${runId}/ws`;

    const ws = new WebSocket(url);
    wsRef.current = ws;
    setConnectionStatus('connecting');

    ws.onopen = () => {
      setConnectionStatus('connected');
      retryRef.current = 0;
    };

    ws.onmessage = (e) => {
      const event = JSON.parse(e.data);
      const seq = event.seq;

      if (seq && seenSeqs.current.has(seq)) return;
      if (seq) seenSeqs.current.add(seq);

      setEvents((prev) => [...prev, event]);

      if (TERMINAL_EVENTS.has(event.event)) {
        setIsTerminal(true);
        setConnectionStatus('disconnected');
        queryClient.invalidateQueries({ queryKey: ['run', runId] });
      }
    };

    ws.onclose = () => {
      if (isTerminal) return;
      setConnectionStatus('disconnected');

      if (retryRef.current < MAX_RETRIES) {
        const delay = RETRY_DELAYS[retryRef.current] || 4000;
        retryRef.current += 1;
        timerRef.current = setTimeout(connect, delay);
      }
    };

    ws.onerror = () => {
      setConnectionStatus('error');
    };
  }, [runId, isTerminal, queryClient]);

  useEffect(() => {
    connect();

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connect]);

  const latestEvent = events[events.length - 1] || null;

  return { events, connectionStatus, isTerminal, latestEvent };
}
