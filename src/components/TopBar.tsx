import React, { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Play, Pause, Square, RotateCcw, Wifi, WifiOff, Film } from 'lucide-react';
import { useStore } from '../store/store';

const API = 'http://localhost:8000';

function formatTime(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return [h, m, s].map((n) => String(n).padStart(2, '0')).join(':');
}

export const TopBar = React.memo(() => {
  const wsConnected  = useStore((s) => s.wsConnected);
  const systemState  = useStore((s) => s.systemState);
  const metrics      = useStore((s) => s.metrics);
  const mode         = useStore((s) => s.mode);
  const setMode      = useStore((s) => s.setMode);
  const applyWSMessage = useStore((s) => s.applyWSMessage);
  const reward       = metrics.reward;
  const isPlaying    = systemState.status === 'PLAYING';

  const [demoRunning, setDemoRunning] = useState(false);
  const demoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Optimistically update local status then hit backend
  const sendControl = useCallback(async (action: 'start' | 'pause' | 'stop' | 'reset') => {
    // Optimistic UI: update local store immediately
    const nextStatus =
      action === 'start' ? 'PLAYING' :
      action === 'pause' ? 'PAUSED'  :
      action === 'stop'  ? 'PAUSED'  : 'PLAYING'; // reset → PLAYING
    applyWSMessage({ system_state: { status: nextStatus as 'PLAYING' | 'PAUSED' | 'STOPPED' } });

    try {
      await fetch(`${API}/control/${action}`, { method: 'POST' });
    } catch {
      // connection error is shown by WS indicator
    }
  }, [applyWSMessage]);

  const setSimulationMode = useCallback(async (m: 'BASELINE' | 'TRAINED') => {
    setMode(m);
    applyWSMessage({ system_state: { mode: m } });
    try {
      await fetch(`${API}/control/mode?mode=${m}`, { method: 'POST' });
    } catch {
      // ignore
    }
  }, [setMode, applyWSMessage]);

  // Demo mode: 12s BASELINE → auto-switch to TRAINED
  const runDemo = useCallback(async () => {
    if (demoRunning) return;
    setDemoRunning(true);

    // Step 1: reset + go BASELINE
    await setSimulationMode('BASELINE');
    await sendControl('reset');

    // Step 2: after 12s switch to TRAINED and show improvements
    demoTimerRef.current = setTimeout(async () => {
      await setSimulationMode('TRAINED');
      setDemoRunning(false);
    }, 12000);
  }, [demoRunning, setSimulationMode, sendControl]);

  const selectScenario = useCallback(async (scenario: 'earthquake' | 'flood' | 'blackout') => {
    // Optimistic update: show scenario name immediately + PLAYING state
    applyWSMessage({ system_state: { scenario: scenario.charAt(0).toUpperCase() + scenario.slice(1), status: 'PLAYING' } });
    try {
      // Backend resets env + auto-starts (simulation_status = "PLAYING")
      await fetch(`${API}/control/scenario?scenario=${scenario}&level=2&seed=42`, { method: 'POST' });
    } catch {
      // ignore
    }
  }, [applyWSMessage]);

  return (
    <div style={{
      position: 'absolute', top: 0, left: 0, right: 0, height: 64,
      zIndex: 60, display: 'flex', alignItems: 'center', gap: 0,
      background: 'rgba(5,10,22,0.95)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid rgba(0,180,255,0.1)',
      boxShadow: '0 2px 30px rgba(0,0,0,0.7)',
      padding: '0 16px',
    }}>

      {/* ── Brand ── */}
      <div style={{ minWidth: 170 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontSize: 17, fontWeight: 800, letterSpacing: '0.04em',
            background: 'linear-gradient(90deg,#00d4ff,#7788ff)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>MetaX Scaler</span>
          {wsConnected
            ? <Wifi size={11} color="#00ff88" />
            : <WifiOff size={11} color="#ff3355" style={{ animation: 'blink 1s step-end infinite' }} />}
          <span style={{ fontSize: 9, color: wsConnected ? '#00ff88' : '#ff3355', letterSpacing: '0.1em' }}>
            {wsConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-secondary)', letterSpacing: '0.08em' }}>Vijayawada City</div>
        <div style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>AI Command Center</div>
      </div>

      <div style={{ width: 1, height: 40, background: 'rgba(80,120,220,0.2)', margin: '0 14px' }} />

      {/* ── Scenario ── */}
      <div style={{ flex: '0 0 auto' }}>
        <div style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 2 }}>Current Scenario</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{systemState.scenario}</span>
          <Activity size={12} color="#ff3355" style={{ animation: 'blink 1s step-end infinite' }} />
        </div>
        <div style={{ fontSize: 8, color: '#ff5566', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>HIGH SEVERITY</div>
      </div>

      <div style={{ flex: 1 }} />

      {/* ── Timer ── */}
      <div style={{ textAlign: 'center', padding: '0 20px' }}>
        <div style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 1 }}>Simulation Time</div>
        <div style={{
          fontSize: 22, fontFamily: 'var(--font-mono)', color: '#00d4ff', fontWeight: 400,
          letterSpacing: '0.08em', lineHeight: 1,
          textShadow: '0 0 12px rgba(0,212,255,0.5)',
        }}>
          {formatTime(systemState.time_elapsed)}
        </div>
        <div style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>Elapsed</div>
      </div>

      <div style={{ width: 1, height: 40, background: 'rgba(80,120,220,0.2)', margin: '0 16px' }} />

      {/* ── Reward ── */}
      <div style={{ textAlign: 'center', padding: '0 16px' }}>
        <div style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 1 }}>Total Reward</div>
        <AnimatePresence mode="wait">
          <motion.div
            key={Math.round(reward / 10)}
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              fontSize: 22, fontFamily: 'var(--font-mono)', fontWeight: 700,
              letterSpacing: '0.06em', lineHeight: 1,
              color: reward >= 0 ? '#00ff88' : '#ff3355',
              textShadow: `0 0 12px ${reward >= 0 ? 'rgba(0,255,136,0.5)' : 'rgba(255,51,85,0.5)'}`,
            }}
          >
            {reward >= 0 ? '+' : ''}{reward.toLocaleString()}
          </motion.div>
        </AnimatePresence>
        <div style={{ fontSize: 8, color: '#00ff88', letterSpacing: '0.12em', textTransform: 'uppercase' }}>+ LIVE</div>
      </div>

      <div style={{ width: 1, height: 40, background: 'rgba(80,120,220,0.2)', margin: '0 16px' }} />

      {/* ── Controls ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <button style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '6px 14px', borderRadius: 7, cursor: 'pointer',
          background: 'rgba(0,220,100,0.18)', color: '#00ff88',
          border: '1px solid rgba(0,220,100,0.35)', fontSize: 11, fontWeight: 700,
          letterSpacing: '0.05em', transition: 'all 0.2s',
          opacity: isPlaying ? 0.65 : 1,
        }} onClick={() => sendControl('start')} disabled={isPlaying}>
          <Play size={12} /> Start
        </button>
        <button style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '6px 12px', borderRadius: 7, cursor: 'pointer',
          background: 'rgba(180,120,0,0.18)', color: '#ffaa00',
          border: '1px solid rgba(180,120,0,0.35)', fontSize: 11, fontWeight: 700,
          letterSpacing: '0.05em',
          opacity: !isPlaying ? 0.65 : 1,
        }} onClick={() => sendControl('pause')} disabled={!isPlaying}>
          <Pause size={12} /> Pause
        </button>
        <button style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '6px 12px', borderRadius: 7, cursor: 'pointer',
          background: 'rgba(120,80,180,0.2)', color: '#c8a8ff',
          border: '1px solid rgba(160,120,220,0.35)', fontSize: 11, fontWeight: 700,
          letterSpacing: '0.05em',
          opacity: !isPlaying ? 0.65 : 1,
        }} onClick={() => sendControl('stop')} disabled={!isPlaying} title="Stop simulation (same as pause)">
          <Square size={10} fill="currentColor" /> Stop
        </button>
        <button style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '6px 12px', borderRadius: 7, cursor: 'pointer',
          background: 'rgba(180,50,50,0.15)', color: '#ff8888',
          border: '1px solid rgba(180,50,50,0.3)', fontSize: 11, fontWeight: 700,
          letterSpacing: '0.05em',
        }} onClick={() => sendControl('reset')}>
          <RotateCcw size={12} /> Reset
        </button>

        {/* Demo Mode */}
        <button
          title="Run 12s Baseline then auto-switch to Trained — shows RL improvement"
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '6px 12px', borderRadius: 7, cursor: demoRunning ? 'not-allowed' : 'pointer',
            background: demoRunning ? 'rgba(187,68,255,0.25)' : 'rgba(187,68,255,0.12)',
            color: demoRunning ? '#e0aaff' : '#cc88ff',
            border: `1px solid ${demoRunning ? 'rgba(187,68,255,0.55)' : 'rgba(187,68,255,0.3)'}`,
            fontSize: 11, fontWeight: 700, letterSpacing: '0.05em',
            boxShadow: demoRunning ? '0 0 14px rgba(187,68,255,0.3)' : 'none',
            transition: 'all 0.2s',
          }}
          onClick={runDemo}
          disabled={demoRunning}
        >
          <Film size={12} />
          {demoRunning ? 'DEMO…' : '🎬 DEMO'}
        </button>
      </div>

      <div style={{ width: 1, height: 40, background: 'rgba(80,120,220,0.2)', margin: '0 14px' }} />

      {/* ── Scenario selector ── */}
      <div style={{ marginRight: 8 }}>
        <div style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', textAlign: 'center', marginBottom: 4 }}>Scenario</div>
        <div style={{
          display: 'flex', background: 'rgba(0,8,22,0.8)',
          border: '1px solid rgba(80,120,220,0.2)', borderRadius: 7, padding: 3, gap: 1,
        }}>
          {(['earthquake', 'flood', 'blackout'] as const).map((s) => {
            const active = systemState.scenario?.toLowerCase().includes(s);
            const colors: Record<string, string> = {
              earthquake: '#ffaa00', flood: '#00d4ff', blackout: '#bb44ff',
            };
            return (
              <button key={s} onClick={() => selectScenario(s)} style={{
                padding: '3px 8px', borderRadius: 5, border: 'none', cursor: 'pointer',
                fontSize: 9, fontWeight: 700, letterSpacing: '0.04em', transition: 'all 0.2s',
                background: active ? `${colors[s]}22` : 'transparent',
                color: active ? colors[s] : 'var(--text-muted)',
                textTransform: 'capitalize',
              }}>{s.slice(0, 3).toUpperCase()}</button>
            );
          })}
        </div>
      </div>

      <div style={{ width: 1, height: 40, background: 'rgba(80,120,220,0.2)', margin: '0 6px' }} />

      {/* ── Mode toggle ── */}
      <div>
        <div style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', textAlign: 'right', marginBottom: 4 }}>Mode</div>
        <div style={{
          display: 'flex', background: 'rgba(0,8,22,0.8)',
          border: '1px solid rgba(80,120,220,0.2)', borderRadius: 7, padding: 3,
        }}>
          {(['BASELINE', 'TRAINED'] as const).map((m) => (
            <button key={m} onClick={() => setSimulationMode(m)} style={{
              padding: '4px 12px', borderRadius: 5, border: 'none', cursor: 'pointer',
              fontSize: 10, fontWeight: 700, letterSpacing: '0.05em', transition: 'all 0.2s',
              background: mode === m
                ? (m === 'TRAINED' ? 'rgba(0,100,255,0.35)' : 'rgba(80,80,100,0.45)')
                : 'transparent',
              color: mode === m ? (m === 'TRAINED' ? '#00d4ff' : '#aab') : 'var(--text-muted)',
              boxShadow: mode === m && m === 'TRAINED' ? '0 0 12px rgba(0,136,255,0.3)' : 'none',
            }}>{m}</button>
          ))}
        </div>
      </div>

      <div style={{ width: 1, height: 40, background: 'rgba(80,120,220,0.2)', margin: '0 8px' }} />

      {/* ── LLM Badge ── */}
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.35)', letterSpacing: '0.1em', marginBottom: 2 }}>
          {mode === 'TRAINED' ? 'ACTIVE LLM' : 'POLICY'}
        </div>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 5,
          background: mode === 'TRAINED' ? 'rgba(0,212,255,0.1)' : 'rgba(120,80,0,0.12)',
          border: `1px solid ${mode === 'TRAINED' ? 'rgba(0,212,255,0.25)' : 'rgba(120,80,0,0.25)'}`,
          borderRadius: 6, padding: '3px 8px',
        }}>
          <div style={{
            width: 5, height: 5, borderRadius: '50%',
            background: mode === 'TRAINED' ? '#00ff88' : '#888',
            boxShadow: mode === 'TRAINED' ? '0 0 6px #00ff88' : 'none',
          }} />
          <span style={{
            fontSize: 9, fontWeight: 800, letterSpacing: '0.04em',
            color: mode === 'TRAINED' ? '#00d4ff' : 'rgba(255,255,255,0.4)',
          }}>
            {mode === 'TRAINED' ? 'LLaMA 3.1-8B' : 'RULE-BASED'}
          </span>
        </div>
        {mode === 'TRAINED' && (
          <div style={{ fontSize: 7, color: 'rgba(0,212,255,0.5)', marginTop: 1 }}>via HuggingFace</div>
        )}
      </div>
    </div>
  );
});
