import { useEffect, useRef } from 'react';
import { useStore } from '../store/store';

function wsUrl(): string {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL as string;
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws`;
}
function snapshotUrl(): string {
  if (import.meta.env.VITE_API_URL) {
    return `${import.meta.env.VITE_API_URL.replace(/\/$/, '')}/snapshot`;
  }
  return `${window.location.origin}/snapshot`;
}
const BASE_RECONNECT_MS = 2000;
const MAX_RECONNECT_MS  = 30000;

export function useWebSocket() {
  const wsRef          = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const attempts       = useRef(0);
  const mounted        = useRef(true);

  const { setWsConnected, setWsError, applyWSMessage } = useStore.getState();

  useEffect(() => {
    mounted.current = true;

    const bootstrapSnapshot = async () => {
      try {
        const res = await fetch(snapshotUrl());
        if (!res.ok) return;
        const data = await res.json();
        if (mounted.current) {
          applyWSMessage(data);
        }
      } catch {
        // snapshot is best-effort; websocket remains source of truth
      }
    };

    const connect = () => {
      if (!mounted.current) return;

      try {
        const url = wsUrl();
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!mounted.current) return;
          attempts.current = 0;
          setWsConnected(true);
          setWsError(null);
          console.info('[WS] Connected →', url);
        };

        ws.onmessage = (ev) => {
          if (!mounted.current) return;
          try {
            const data = JSON.parse(ev.data as string);
            applyWSMessage(data);
          } catch {
            // malformed frame — skip
          }
        };

        ws.onerror = () => {
          if (!mounted.current) return;
          setWsError('WebSocket error');
        };

        ws.onclose = () => {
          if (!mounted.current) return;
          setWsConnected(false);
          attempts.current++;
          const delay = Math.min(BASE_RECONNECT_MS * 2 ** (attempts.current - 1), MAX_RECONNECT_MS);
          console.warn(`[WS] Disconnected. Reconnecting in ${delay}ms (attempt ${attempts.current})`);
          reconnectTimer.current = setTimeout(connect, delay);
        };
      } catch (err) {
        setWsError('Cannot create WebSocket');
        reconnectTimer.current = setTimeout(connect, BASE_RECONNECT_MS);
      }
    };

    bootstrapSnapshot();
    connect();

    return () => {
      mounted.current = false;
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // suppress reconnect on intentional unmount
        wsRef.current.close();
      }
    };
  }, []);
}
