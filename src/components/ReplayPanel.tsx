/**
 * ReplayPanel.tsx
 *
 * Training Replay Mode — visually demonstrates measurable learning.
 *
 * Shows three episodes side-by-side:
 *   Episode 0 (BEFORE): baseline policy — wandering, misinformation, high deaths
 *   Episode 1 (MID):    partial training — mixed behavior, some rescues
 *   Episode 2 (AFTER):  trained policy  — fast rescue, coordination, low deaths
 *
 * Data fetched from GET /replay on the backend.
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';
import { Play, Loader, TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp } from 'lucide-react';

const API = 'http://localhost:8000';

interface EpisodeSummary {
  total_reward: number;
  deaths: number;
  panic: number;
  trust: number;
  rescue_rate: number;
  coordination_score: number;
}

interface EpisodeStep {
  step: number;
  reward: number;
  deaths: number;
  panic: number;
  trust: number;
  lives_saved: number;
  rescue_rate: number;
}

interface Episode {
  policy: string;
  steps: EpisodeStep[];
  summary: EpisodeSummary;
}

interface ReplayData {
  episodes: Episode[];
  delta: {
    reward_gain: number;
    deaths_reduced: number;
    panic_reduction: number;
    rescue_rate_gain: number;
    trust_improvement: number;
  };
}

const PHASE_COLORS = ['#ff3355', '#ffaa00', '#00ff88'];
const PHASE_BG = [
  'rgba(255,51,85,0.08)',
  'rgba(255,170,0,0.08)',
  'rgba(0,255,136,0.08)',
];
const PHASE_BORDER = [
  'rgba(255,51,85,0.3)',
  'rgba(255,170,0,0.3)',
  'rgba(0,255,136,0.3)',
];

function DeltaBadge({ value, unit = '', inverted = false }: { value: number; unit?: string; inverted?: boolean }) {
  const positive = inverted ? value < 0 : value > 0;
  const color = positive ? '#00ff88' : value === 0 ? '#888' : '#ff3355';
  const Icon = positive ? TrendingUp : value === 0 ? Minus : TrendingDown;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, color, fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: 11 }}>
      <Icon size={9} />
      {value > 0 ? '+' : ''}{value}{unit}
    </span>
  );
}

function EpisodeCard({ ep, idx, isActive }: { ep: Episode; idx: number; isActive: boolean }) {
  const color = PHASE_COLORS[idx];
  const s = ep.summary;
  const chartData = ep.steps.map((st) => ({ step: st.step, reward: st.reward, panic: st.panic, lives: st.lives_saved }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: idx * 0.12 }}
      style={{
        flex: 1,
        minWidth: 0,
        background: PHASE_BG[idx],
        border: `1px solid ${PHASE_BORDER[idx]}`,
        borderRadius: 10,
        padding: '10px 12px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Glow pulse for active/trained */}
      {isActive && (
        <motion.div
          animate={{ opacity: [0.15, 0.4, 0.15] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{ position: 'absolute', inset: 0, background: `radial-gradient(ellipse at 50% 0%, ${color}22, transparent 70%)`, pointerEvents: 'none' }}
        />
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        <div style={{ width: 6, height: 6, borderRadius: '50%', background: color, boxShadow: `0 0 6px ${color}` }} />
        <span style={{ fontSize: 9, fontWeight: 900, letterSpacing: '0.1em', color, textTransform: 'uppercase' }}>
          {ep.policy}
        </span>
      </div>

      {/* Key metrics grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 10px', marginBottom: 8 }}>
        {[
          { label: 'TOTAL REWARD', val: s.total_reward.toLocaleString(), color: s.total_reward >= 0 ? '#00ff88' : '#ff3355' },
          { label: 'DEATHS', val: String(s.deaths), color: s.deaths === 0 ? '#00ff88' : s.deaths < 3 ? '#ffaa00' : '#ff3355' },
          { label: 'PANIC', val: `${s.panic}%`, color: s.panic < 20 ? '#00ff88' : s.panic < 50 ? '#ffaa00' : '#ff3355' },
          { label: 'RESCUE RATE', val: `${s.rescue_rate}%`, color: s.rescue_rate > 60 ? '#00ff88' : s.rescue_rate > 30 ? '#ffaa00' : '#ff3355' },
          { label: 'TRUST', val: `${s.trust}`, color: s.trust > 70 ? '#00ff88' : s.trust > 40 ? '#ffaa00' : '#ff3355' },
          { label: 'COORD SCORE', val: `${s.coordination_score.toFixed(0)}`, color: color },
        ].map(({ label, val, color: vc }) => (
          <div key={label}>
            <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.06em' }}>{label}</div>
            <div style={{ fontSize: 13, fontWeight: 800, color: vc, fontFamily: 'var(--font-mono)', lineHeight: 1.2 }}>{val}</div>
          </div>
        ))}
      </div>

      {/* Reward curve */}
      {chartData.length > 0 && (
        <div style={{ height: 36 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={`rg${idx}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.4} />
                  <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="reward"
                stroke={color}
                strokeWidth={1.5}
                fill={`url(#rg${idx})`}
                isAnimationActive={true}
                animationDuration={800}
                dot={false}
              />
              <Tooltip
                contentStyle={{ background: '#0a1228', border: `1px solid ${color}44`, fontSize: 9, padding: '3px 6px' }}
                labelStyle={{ color: 'rgba(255,255,255,0.5)', fontSize: 8 }}
                itemStyle={{ color }}
                formatter={(v: number) => [v.toFixed(1), 'reward']}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </motion.div>
  );
}

export const ReplayPanel = React.memo(() => {
  const [data, setData] = useState<ReplayData | null>(null);
  const [loading, setLoading] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const hasFetched = useRef(false);

  const fetchReplay = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/replay?level=2&steps=30`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch {
      // backend unavailable
    } finally {
      setLoading(false);
    }
  };

  // Auto-fetch once on mount (best-effort)
  useEffect(() => {
    if (!hasFetched.current) {
      hasFetched.current = true;
      fetchReplay();
    }
  }, []);

  return (
    <div
      className="glass"
      style={{
        position: 'absolute',
        bottom: 10,
        left: '50%',
        transform: 'translateX(-50%)',
        width: 'min(900px, calc(100vw - 360px))',
        zIndex: 50,
        borderRadius: 12,
        overflow: 'hidden',
        pointerEvents: 'auto',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '8px 14px',
          background: 'rgba(0,0,0,0.4)',
          borderBottom: '1px solid rgba(0,180,255,0.12)',
          cursor: 'pointer',
        }}
        onClick={() => setCollapsed((c) => !c)}
      >
        <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#00ff88', boxShadow: '0 0 6px #00ff88' }} />
        <span style={{ fontSize: 10, fontWeight: 900, letterSpacing: '0.15em', textTransform: 'uppercase', color: '#fff' }}>
          Training Replay — Before / Mid / After
        </span>
        <span style={{ fontSize: 8, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.08em', marginLeft: 4 }}>
          LLMs learning coordination under uncertainty
        </span>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          {data && (
            <span style={{ fontSize: 8, color: '#00ff88', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
              Reward +{data.delta.reward_gain > 0 ? '+' : ''}{data.delta.reward_gain} · Deaths -{data.delta.deaths_reduced} · Rescue +{data.delta.rescue_rate_gain}%
            </span>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); fetchReplay(); }}
            disabled={loading}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '3px 10px', borderRadius: 5, border: '1px solid rgba(0,212,255,0.3)',
              background: 'rgba(0,212,255,0.1)', color: '#00d4ff', fontSize: 9, fontWeight: 700,
              cursor: loading ? 'default' : 'pointer', opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? <Loader size={9} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={9} />}
            {loading ? 'Running…' : 'Run Replay'}
          </button>
          {collapsed ? <ChevronUp size={12} color="rgba(255,255,255,0.4)" /> : <ChevronDown size={12} color="rgba(255,255,255,0.4)" />}
        </div>
      </div>

      {/* Body */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <div style={{ padding: '10px 12px' }}>
              {!data && !loading && (
                <div style={{ textAlign: 'center', padding: '14px 0', color: 'rgba(255,255,255,0.35)', fontSize: 11 }}>
                  Press Run Replay to compare baseline vs trained behavior.
                </div>
              )}

              {loading && (
                <div style={{ textAlign: 'center', padding: '14px 0', color: '#00d4ff', fontSize: 11 }}>
                  Running 3 episodes… (≈15s)
                </div>
              )}

              {data && !loading && (
                <>
                  {/* Episode cards */}
                  <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
                    {data.episodes.map((ep, idx) => (
                      <EpisodeCard key={idx} ep={ep} idx={idx} isActive={idx === 2} />
                    ))}
                  </div>

                  {/* Delta summary bar */}
                  <div style={{
                    display: 'flex', gap: 12, padding: '7px 12px',
                    background: 'rgba(0,255,136,0.05)',
                    border: '1px solid rgba(0,255,136,0.15)',
                    borderRadius: 7, alignItems: 'center',
                  }}>
                    <span style={{ fontSize: 8, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.1em', fontWeight: 700 }}>IMPROVEMENT</span>
                    {[
                      { label: 'Reward', value: data.delta.reward_gain, unit: '' },
                      { label: 'Deaths', value: -data.delta.deaths_reduced, unit: '', inverted: true },
                      { label: 'Panic', value: -data.delta.panic_reduction, unit: '%', inverted: true },
                      { label: 'Rescue', value: data.delta.rescue_rate_gain, unit: '%' },
                      { label: 'Trust', value: data.delta.trust_improvement, unit: '' },
                    ].map(({ label, value, unit, inverted }) => (
                      <div key={label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <span style={{ fontSize: 7, color: 'rgba(255,255,255,0.35)', letterSpacing: '0.06em' }}>{label}</span>
                        <DeltaBadge value={value} unit={unit} inverted={inverted} />
                      </div>
                    ))}
                    <div style={{ marginLeft: 'auto', fontSize: 8, color: 'rgba(255,255,255,0.45)', maxWidth: 240, textAlign: 'right', lineHeight: 1.4 }}>
                      Same scenario · Same seed · Only policy differs.
                      Demonstrates measurable RL improvement.
                    </div>
                  </div>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});
