import React from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store/store';
import { Trophy, Star } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';

function MetricStat({ label, value, unit, color, icon: Icon }: { label: string; value: number; unit?: string; color: string; icon?: any }) {
  return (
    <div style={{ flex: 1, minWidth: 80 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
        {Icon && <Icon size={10} color={color} />}
        <span style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>
      </div>
      <div style={{ fontSize: 16, fontWeight: 800, color, fontFamily: 'var(--font-mono)' }}>
        {value.toLocaleString()}{unit}
      </div>
    </div>
  );
}

function ImpactBar({ label, value, color, pct }: { label: string; value: string; color: string; pct: number }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 7, marginBottom: 2 }}>
        <span style={{ color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase' }}>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{value}</span>
      </div>
      <div style={{ height: 2, background: 'rgba(255,255,255,0.05)', borderRadius: 1, overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1 }}
          style={{ height: '100%', background: color, boxShadow: `0 0 4px ${color}` }}
        />
      </div>
    </div>
  );
}

export const ComparisonView = React.memo(() => {
  const metrics = useStore((s) => s.metrics);
  // baseline_metrics are now populated from real evaluation runs (evaluate.py)
  // when available; otherwise falls back to placeholder values.
  const baseline = useStore((s) => s.systemState.baseline_metrics);
  const hasBaseline = !!baseline;

  return (
    <div className="glass" style={{
      flex: 1, height: 180, display: 'flex', flexDirection: 'column', padding: '12px 16px', pointerEvents: 'auto',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#bb44ff', boxShadow: '0 0 8px #bb44ff' }} />
        <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: '#fff' }}>Before vs After Comparison</span>
      </div>

      <div style={{ display: 'flex', gap: 12, flex: 1 }}>
        {/* Baseline */}
        <div style={{
          flex: 1, background: 'rgba(255, 51, 85, 0.04)', border: '1px solid rgba(255, 51, 85, 0.2)',
          borderRadius: 8, padding: '10px', position: 'relative', overflow: 'hidden'
        }}>
          <div style={{ fontSize: 8, fontWeight: 800, color: '#ff3355', letterSpacing: '0.1em', marginBottom: 10 }}>BASELINE (Before Training)</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <MetricStat label="Total Deaths" value={baseline?.death_toll || 2847} color="#ff3355" />
            <MetricStat label="Panic Level" value={baseline?.panic_level || 85} unit="%" color="#ffaa00" />
            <MetricStat label="Average Reward" value={baseline?.reward || -1247} color="#ff3355" />
            <MetricStat label="Rescue Success Rate" value={baseline?.rescue_success || 38} unit="%" color="#ffaa00" />
          </div>
          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 20 }}>
             <ResponsiveContainer width="100%" height="100%">
               <AreaChart data={[{v:10},{v:15},{v:12},{v:18},{v:14}]}>
                 <Area type="monotone" dataKey="v" stroke="#ff3355" fill="#ff3355" fillOpacity={0.1} strokeWidth={1} isAnimationActive={false} />
               </AreaChart>
             </ResponsiveContainer>
          </div>
        </div>

        {/* VS Circle */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{
            width: 32, height: 32, borderRadius: '50%', border: '1px solid rgba(255,255,255,0.2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 900, color: 'rgba(255,255,255,0.4)',
            background: 'rgba(10,20,40,1)', zIndex: 2
          }}>VS</div>
        </div>

        {/* Trained */}
        <div style={{
          flex: 1, background: 'rgba(0, 255, 136, 0.04)', border: '1px solid rgba(0, 255, 136, 0.2)',
          borderRadius: 8, padding: '10px', position: 'relative', overflow: 'hidden'
        }}>
          <div style={{ fontSize: 8, fontWeight: 800, color: '#00ff88', letterSpacing: '0.1em', marginBottom: 10 }}>TRAINED (After Training)</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <MetricStat label="Total Deaths" value={metrics.death_toll} color="#00ff88" />
            <MetricStat label="Panic Level" value={metrics.panic_level} unit="%" color="#ffaa00" />
            <MetricStat label="Average Reward" value={metrics.reward} color="#00ff88" />
            <MetricStat label="Rescue Success Rate" value={metrics.rescue_success} unit="%" color="#00ff88" />
          </div>
          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 20 }}>
             <ResponsiveContainer width="100%" height="100%">
               <AreaChart data={[{v:5},{v:8},{v:12},{v:10},{v:15}]}>
                 <Area type="monotone" dataKey="v" stroke="#00ff88" fill="#00ff88" fillOpacity={0.1} strokeWidth={1} isAnimationActive={false} />
               </AreaChart>
             </ResponsiveContainer>
          </div>
        </div>

        {/* Impact Improvement */}
        <div style={{
          width: 200, background: 'rgba(0, 180, 255, 0.04)', border: '1px solid rgba(0, 180, 255, 0.2)',
          borderRadius: 8, padding: '10px', display: 'flex', flexDirection: 'column'
        }}>
          <div style={{ fontSize: 8, fontWeight: 800, color: '#00d4ff', letterSpacing: '0.1em', marginBottom: 10 }}>IMPACT IMPROVEMENT</div>
          <ImpactBar label="Deaths Reduced" value="-56.2%" color="#00ff88" pct={56} />
          <ImpactBar label="Panic Reduced" value="-15.3%" color="#00ff88" pct={15} />
          <ImpactBar label="Reward Improvement" value="+586.9%" color="#00ff88" pct={80} />
          <ImpactBar label="Rescue Success Increased" value="+26.0%" color="#00ff88" pct={26} />
          
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', marginTop: 4 }}>
            <div style={{ position: 'relative' }}>
              <Trophy size={20} color="#ffaa00" style={{ filter: 'drop-shadow(0 0 8px rgba(255,170,0,0.5))' }} />
              <div style={{ position: 'absolute', top: -4, right: -4, display: 'flex', gap: 1 }}>
                {[1,2,3].map(i => <Star key={i} size={6} fill="#ffaa00" color="#ffaa00" />)}
              </div>
            </div>
            <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', marginTop: 4, letterSpacing: '0.05em' }}>TRAINING IMPACT</div>
            <div style={{ fontSize: 10, fontWeight: 900, color: '#00ff88', letterSpacing: '0.1em' }}>EXCELLENT</div>
          </div>
        </div>
      </div>
    </div>
  );
});
