import React, { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore, EventLog, Severity } from '../store/store';

const SEV_COLOR: Record<Severity, string> = {
  CRITICAL: '#bb44ff',
  HIGH:     '#ff3355',
  MEDIUM:   '#ffaa00',
  LOW:      '#00ff88',
};

function EventRow({ ev }: { ev: EventLog }) {
  const color = SEV_COLOR[ev.severity] ?? 'rgba(255,255,255,0.4)';
  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0 }}
      style={{
        display: 'grid',
        gridTemplateColumns: '60px 1fr 45px',
        gap: '0 10px',
        padding: '6px 0',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        alignItems: 'center',
      }}
    >
      <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.35)', fontFamily: 'var(--font-mono)' }}>
        {ev.timestamp}
      </span>
      <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.85)', lineHeight: 1.3 }}>
        {ev.message}
      </span>
      <span style={{
        fontSize: 8, fontWeight: 800, letterSpacing: '0.04em', textTransform: 'uppercase',
        color, textAlign: 'right',
      }}>
        {ev.severity}
      </span>
    </motion.div>
  );
}

export const Timeline = React.memo(() => {
  const events = useStore((s) => s.events);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [events.length]);

  return (
    <div
      className="glass"
      style={{
        width: 320,
        height: 180,
        display: 'flex',
        flexDirection: 'column',
        padding: '12px 14px',
        pointerEvents: 'auto',
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexShrink: 0 }}>
        <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#ffaa00', boxShadow: '0 0 8px #ffaa00' }} />
        <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: '#fff' }}>
          Events · decisions
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 9, color: 'rgba(255,255,255,0.3)' }}>
          {events.length} events
        </span>
      </div>

      <div
        ref={listRef}
        style={{ overflowY: 'auto', flex: 1, paddingRight: 4 }}
        className="custom-scrollbar"
      >
        {events.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '20px 0', color: 'rgba(255,255,255,0.2)', fontSize: 11 }}>
            Awaiting event stream…
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {events.map((ev) => (
              <EventRow key={ev.id} ev={ev} />
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
});
