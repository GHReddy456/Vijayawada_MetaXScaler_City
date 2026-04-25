import { useEffect, useRef } from 'react';
import { useStore } from '../store/store';

const WS_URL = 'ws://localhost:8000/ws';
const HTTP_SNAPSHOT_URL = 'http://localhost:8000/snapshot';
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
        const res = await fetch(HTTP_SNAPSHOT_URL);
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
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!mounted.current) return;
          attempts.current = 0;
          setWsConnected(true);
          setWsError(null);
          console.info('[WS] Connected →', WS_URL);
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
