/**
 * AgentChat — Live Agent Chat Timeline
 *
 * Narrates every simulation step as a scrollable, colour-coded log so that
 * a judge can follow agent reasoning, communication, and improvement purely
 * by watching this panel.
 *
 * Entry colours:
 *   event    → #3b9eff  (blue)
 *   decision → #22c55e  (green)
 *   message  → #a855f7  (purple)
 *   outcome  → #f97316  (orange)
 */

import React, { useEffect, useRef } from 'react';
import { useStore } from '../store/store';
import type { ChatEvent, ChatEventType } from '../store/store';

// ── constants ────────────────────────────────────────────────────────────────

const COLORS: Record<ChatEventType, string> = {
  event:    '#3b9eff',
  decision: '#22c55e',
  message:  '#a855f7',
  outcome:  '#f97316',
};

const BG: Record<ChatEventType, string> = {
  event:    'rgba(59,158,255,0.08)',
  decision: 'rgba(34,197,94,0.08)',
  message:  'rgba(168,85,247,0.08)',
  outcome:  'rgba(249,115,22,0.08)',
};

const ICONS: Record<ChatEventType, string> = {
  event:    '📍',
  decision: '⚙️',
  message:  '💬',
  outcome:  '📊',
};

const LABELS: Record<ChatEventType, string> = {
  event:    'EVENT',
  decision: 'DECISION',
  message:  'MESSAGE',
  outcome:  'OUTCOME',
};

// ── sub-components ────────────────────────────────────────────────────────────

interface RowProps {
  entry: ChatEvent;
  idx: number;
}

function ChatRow({ entry, idx }: RowProps) {
  const { type, text, ts } = entry;
  const color  = COLORS[type] ?? '#9ca3af';
  const bg     = BG[type]    ?? 'rgba(156,163,175,0.06)';
  const icon   = ICONS[type] ?? '•';
  const label  = LABELS[type] ?? type.toUpperCase();

  return (
    <div
      key={idx}
      style={{
        display: 'flex',
        gap: 8,
        alignItems: 'flex-start',
        padding: '6px 10px',
        background: bg,
        borderLeft: `2px solid ${color}`,
        borderRadius: 4,
        marginBottom: 4,
        animation: 'chatFadeIn 0.25s ease-out both',
      }}
    >
      {/* icon + label */}
      <span style={{ fontSize: 13, lineHeight: '18px', minWidth: 18 }}>{icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <span
          style={{
            fontSize: 9,
            fontWeight: 700,
            color,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            marginRight: 6,
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontSize: 11.5,
            color: '#d1d5db',
            lineHeight: '16px',
            wordBreak: 'break-word',
          }}
        >
          {text}
        </span>
      </div>
      {/* timestamp */}
      {ts && (
        <span
          style={{
            fontSize: 9,
            color: '#4b5563',
            whiteSpace: 'nowrap',
            marginTop: 2,
          }}
        >
          {ts}
        </span>
      )}
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export function AgentChat() {
  const chatEvents = useStore((s) => s.chatEvents);
  const status     = useStore((s) => s.systemState.status);
  const scrollRef  = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever new events arrive
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Only auto-scroll if user is already near bottom (within 120 px)
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) {
      el.scrollTop = el.scrollHeight;
    }
  }, [chatEvents.length]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: 340,
        height: '100%',
        background: 'rgba(6,13,26,0.96)',
        border: '1px solid rgba(59,158,255,0.18)',
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      {/* ── header ── */}
      <div
        style={{
          padding: '8px 12px',
          background: 'rgba(59,158,255,0.07)',
          borderBottom: '1px solid rgba(59,158,255,0.15)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 700, color: '#3b9eff', letterSpacing: '0.12em' }}>
          🧠 AGENT CHAT TIMELINE
        </span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {/* live pulse when running */}
          {status === 'PLAYING' && (
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: '#22c55e',
                animation: 'pulse 1.1s infinite',
                display: 'inline-block',
              }}
            />
          )}
          <span style={{ fontSize: 9, color: '#6b7280' }}>
            {chatEvents.length} entries
          </span>
        </div>
      </div>

      {/* ── legend ── */}
      <div
        style={{
          display: 'flex',
          gap: 10,
          padding: '4px 12px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          flexShrink: 0,
          flexWrap: 'wrap',
        }}
      >
        {(Object.keys(ICONS) as ChatEventType[]).map((t) => (
          <span key={t} style={{ fontSize: 9, color: COLORS[t], display: 'flex', gap: 3, alignItems: 'center' }}>
            {ICONS[t]} {t}
          </span>
        ))}
      </div>

      {/* ── scrollable body ── */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding: '8px 8px 8px 8px',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(59,158,255,0.25) transparent',
        }}
      >
        {chatEvents.length === 0 ? (
          <div
            style={{
              textAlign: 'center',
              color: '#374151',
              fontSize: 12,
              marginTop: 40,
              lineHeight: '22px',
            }}
          >
            🕐 Waiting for simulation…
            <br />
            <span style={{ fontSize: 10 }}>Press START to see the agent chat</span>
          </div>
        ) : (
          chatEvents.map((e, i) => <ChatRow key={i} entry={e} idx={i} />)
        )}
      </div>

      {/* ── keyframes injected once ── */}
      <style>{`
        @keyframes chatFadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
