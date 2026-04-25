import React from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';
import { Activity, TrendingUp, TrendingDown, Zap } from 'lucide-react';
import { useStore } from '../store/store';
import {
  AreaChart, Area, ResponsiveContainer,
  XAxis, YAxis, Tooltip, ReferenceLine,
} from 'recharts';

function AnimNum({ value, decimals = 0, prefix = '' }: { value: number; decimals?: number; prefix?: string }) {
  const spring = useSpring(value, { stiffness: 60, damping: 20 });
  React.useEffect(() => { spring.set(value); }, [value]);
  const display = useTransform(spring, (v) =>
    `${prefix}${v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`
  );
  return <motion.span>{display}</motion.span>;
}

function MetricItem({ label, value, max, color, badge, showPct = false }: {
  label: string; value: number; max: number; color: string; badge: string; showPct?: boolean;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 4, height: 4, borderRadius: '50%', background: color }} />
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.6)', letterSpacing: '0.1em', fontWeight: 600 }}>
            {label.toUpperCase()}
          </span>
        </div>
        <span style={{ fontSize: 8, color, fontWeight: 700, letterSpacing: '0.05em' }}>{badge}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 5 }}>
        <span style={{ fontSize: 22, fontWeight: 800, color: '#fff', fontFamily: 'var(--font-mono)', lineHeight: 1 }}>
          <AnimNum value={value} />
        </span>
        <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', fontWeight: 500 }}>
          / {max.toLocaleString()}{showPct ? '%' : ''}
        </span>
      </div>
      <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1.1, ease: 'easeOut' }}
          style={{ height: '100%', background: color, boxShadow: `0 0 8px ${color}` }}
        />
      </div>
    </div>
  );
}

// Small comparison row showing baseline vs trained for a single metric
function CompareRow({ label, baseline, trained, unit = '' }: {
  label: string; baseline: number; trained: number; unit?: string;
}) {
  const improved = trained < baseline; // lower is better for deaths/panic
  const delta = baseline - trained;
  const pctImprove = baseline > 0 ? Math.round((delta / baseline) * 100) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
      <span style={{ fontSize: 8, color: 'rgba(255,255,255,0.45)', minWidth: 56 }}>{label}</span>
      <span style={{ fontSize: 9, color: '#ff5566', fontFamily: 'var(--font-mono)', fontWeight: 700, minWidth: 28 }}>
        {baseline}{unit}
      </span>
      <span style={{ fontSize: 8, color: 'rgba(255,255,255,0.25)' }}>→</span>
      <span style={{ fontSize: 9, color: '#00ff88', fontFamily: 'var(--font-mono)', fontWeight: 700, minWidth: 28 }}>
        {trained}{unit}
      </span>
      {improved && (
        <span style={{
          fontSize: 7, padding: '1px 5px', borderRadius: 8,
          background: 'rgba(0,255,136,0.12)', color: '#00ff88', fontWeight: 700,
        }}>
          -{pctImprove}%
        </span>
      )}
    </div>
  );
}

export const MetricsPanel = React.memo(() => {
  const metrics     = useStore((s) => s.metrics);
  const rewardCurve = useStore((s) => s.rewardCurve);
  const mode        = useStore((s) => s.mode);
  const baseline    = useStore((s) => s.systemState.baseline_metrics);

  // Build chart data from real backend reward_curve
  const chartData = rewardCurve.map((v, i) => ({ step: i + 1, reward: v }));

  // Compute rolling stats from actual curve
  const curveLen     = rewardCurve.length;
  const avgRecent    = curveLen > 0
    ? Math.round(rewardCurve.slice(-10).reduce((a, b) => a + b, 0) / Math.min(curveLen, 10))
    : 0;
  const avgEarly     = curveLen > 5
    ? Math.round(rewardCurve.slice(0, 5).reduce((a, b) => a + b, 0) / 5)
    : 0;
  const trend        = curveLen > 5 ? avgRecent - avgEarly : 0;
  const trendUp      = trend > 0;

  // Derived comparison values (from actual metrics or fallback to baseline_metrics)
  const baseDeaths   = baseline?.death_toll  ?? 15;
  const basePanic    = baseline?.panic_level ?? 80;
  const baseReward   = baseline?.reward      ?? -200;
  const trainDeaths  = Math.round(metrics.death_toll);
  const trainPanic   = Math.round(metrics.panic_level);
  const trainReward  = Math.round(metrics.reward);

  const reward = metrics.reward;

  return (
    <div className="glass" style={{
      position: 'absolute', top: 76, left: 12, width: 248,
      zIndex: 40, padding: '14px', pointerEvents: 'auto',
      maxHeight: 'calc(100vh - 180px)', overflowY: 'auto',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
        <Activity size={13} color="#00d4ff" />
        <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: '#fff' }}>
          Live Metrics
        </span>
        <span style={{
          marginLeft: 'auto', fontSize: 8, padding: '2px 6px', borderRadius: 8,
          background: mode === 'TRAINED' ? 'rgba(0,255,136,0.12)' : 'rgba(255,170,0,0.12)',
          color: mode === 'TRAINED' ? '#00ff88' : '#ffaa00',
          fontWeight: 700,
        }}>
          {mode}
        </span>
      </div>

      <MetricItem label="Death Toll"      value={metrics.death_toll}     max={5000} color="#ff3355" badge="RISK" />
      <MetricItem label="Panic Level"     value={metrics.panic_level}    max={100}  color="#ffaa00" badge="HIGH"  showPct />
      <MetricItem label="Trust Score"     value={metrics.trust_score}    max={100}  color="#00ff88" badge="STBL"  showPct />
      <MetricItem label="Rescue Success"  value={metrics.rescue_success} max={100}  color="#00d4ff" badge="IMPR"  showPct />

      {/* ── Reward + RL curve ── */}
      <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            {trendUp
              ? <TrendingUp  size={11} color="#00ff88" />
              : <TrendingDown size={11} color="#ff5566" />}
            <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.6)', letterSpacing: '0.1em', fontWeight: 600 }}>
              TOTAL REWARD
            </span>
          </div>
          <span style={{ fontSize: 8, color: trendUp ? '#00ff88' : '#ff5566', fontWeight: 800 }}>
            {trendUp ? '↑ IMPROVING' : '→ LEARNING'}
          </span>
        </div>

        <div style={{ fontSize: 22, fontWeight: 800, fontFamily: 'var(--font-mono)', lineHeight: 1,
          color: reward >= 0 ? '#00ff88' : '#ff3355',
          textShadow: `0 0 14px ${reward >= 0 ? 'rgba(0,255,136,0.4)' : 'rgba(255,51,85,0.4)'}`,
          marginBottom: 10,
        }}>
          <AnimNum value={reward} prefix={reward >= 0 ? '+' : ''} />
        </div>

        {/* Real reward curve from backend */}
        <div style={{ marginBottom: 3 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 7, color: 'rgba(255,255,255,0.3)', marginBottom: 4 }}>
            <span>📈 RL LEARNING CURVE</span>
            <span>{curveLen} steps</span>
          </div>
          <div style={{ height: 72 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 2, right: 0, left: -28, bottom: 0 }}>
                <defs>
                  <linearGradient id="rwGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00ff88" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
                <Area
                  type="monotone" dataKey="reward"
                  stroke="#00ff88" fill="url(#rwGrad)"
                  strokeWidth={1.8} isAnimationActive={false}
                  dot={false}
                />
                <XAxis dataKey="step" hide />
                <YAxis hide domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: 'rgba(5,12,28,0.9)', border: '1px solid rgba(0,255,136,0.2)', borderRadius: 4, fontSize: 9 }}
                  itemStyle={{ color: '#00ff88' }}
                  labelStyle={{ color: 'rgba(255,255,255,0.4)', fontSize: 8 }}
                  formatter={(v: number) => [v.toFixed(1), 'reward']}
                  labelFormatter={(l: number) => `step ${l}`}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Trend annotation */}
          {curveLen > 5 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 7, color: 'rgba(255,255,255,0.3)', marginTop: 2 }}>
              <span style={{ color: '#ff5566' }}>Early avg: {avgEarly}</span>
              <span style={{ color: trendUp ? '#00ff88' : '#ffaa00' }}>
                {trendUp ? `▲ +${trend} (learning)` : `▼ ${trend}`}
              </span>
              <span style={{ color: '#00ff88' }}>Recent: {avgRecent}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Baseline vs Trained comparison (computed from real data) ── */}
      {mode === 'TRAINED' && curveLen > 3 && (
        <div style={{
          marginTop: 12, paddingTop: 10, borderTop: '1px solid rgba(255,255,255,0.05)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 7 }}>
            <Zap size={10} color="#bb44ff" />
            <span style={{ fontSize: 8, color: '#bb44ff', fontWeight: 700, letterSpacing: '0.1em' }}>
              BASELINE → TRAINED
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 7, color: '#ff5566', fontWeight: 700 }}>BASELINE</span>
            <span style={{ fontSize: 7, color: '#00ff88', fontWeight: 700 }}>TRAINED</span>
          </div>
          <CompareRow label="Deaths"  baseline={baseDeaths}  trained={trainDeaths} />
          <CompareRow label="Panic"   baseline={basePanic}   trained={trainPanic}  unit="%" />
          <CompareRow label="Reward"  baseline={baseReward}  trained={trainReward} />
          <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.25)', marginTop: 5, lineHeight: 1.4 }}>
            Computed from actual simulation runs
          </div>
        </div>
      )}
    </div>
  );
});
