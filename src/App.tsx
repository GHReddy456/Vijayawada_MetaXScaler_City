import React from 'react';
import { useWebSocket }        from './hooks/useWebSocket';
import { TopBar }              from './components/TopBar';
import { CityMap }             from './components/CityMap';
import { MetricsPanel }        from './components/MetricsPanel';
import { AgentPanel }          from './components/AgentPanel';
import { AgentChat }           from './components/AgentChat';
import { Timeline }            from './components/Timeline';
import { ComparisonView }      from './components/ComparisonView';
import { ReplayPanel }         from './components/ReplayPanel';
import { CapabilityBanner }    from './components/CapabilityBanner';

export default function App() {
  useWebSocket();

  return (
    <div style={{
      position: 'relative',
      width: '100vw', height: '100vh',
      overflow: 'hidden',
      background: '#060d1a',
      userSelect: 'none',
      fontFamily: 'Inter, sans-serif',
    }}>
      {/* ── Layer 0: Real Leaflet Vijayawada Map ── */}
      <CityMap />

      {/* ── Layer 1: Top bar ── */}
      <TopBar />

      {/* ── Layer 2: Left — Live Metrics ── */}
      <MetricsPanel />

      {/* ── Layer 2b: Left below metrics — System Capability Claim ── */}
      <CapabilityBanner />

      {/* ── Layer 3: Right — Agent Intelligence ── */}
      <AgentPanel />

      {/* ── Layer 3b: Chat Timeline — sits immediately left of AgentPanel (right:12, width:320) */}
      <div style={{
        position: 'absolute',
        top: 76,
        right: 344,          /* 12 (AgentPanel margin) + 320 (AgentPanel width) + 12 (gap) */
        width: 340,
        height: 'calc(100vh - 180px)',
        zIndex: 45,
        pointerEvents: 'auto',
      }}>
        <AgentChat />
      </div>

      {/* ── Layer 4: Bottom — Timeline + Before/After Comparison ── */}
      <div style={{
        position: 'absolute',
        bottom: 10, left: 10, right: 10,
        display: 'flex', gap: 10,
        zIndex: 40, pointerEvents: 'none',
      }}>
        <Timeline />
        <ComparisonView />
      </div>

      {/* ── Layer 5: Training Replay (centred above bottom strip) ── */}
      <ReplayPanel />
    </div>
  );
}
