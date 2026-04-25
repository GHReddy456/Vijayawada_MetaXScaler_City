/**
 * CapabilityBanner.tsx
 *
 * Explicit capability claim panel — displayed in the top-right corner.
 * States what this system demonstrates to judges and evaluators.
 *
 * Claim: "LLMs can learn coordination and decision-making
 *         under uncertainty via reinforcement learning
 *         in a multi-agent environment."
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../store/store';
import { Brain, Shield, Zap, GitBranch, Eye, ChevronDown, ChevronUp } from 'lucide-react';

const CAPABILITIES = [
  {
    icon: Brain,
    color: '#00d4ff',
    title: 'Partial Observability',
    desc: 'Each agent operates on role-scoped knowledge. Commander sees all; others see local zone + messages only.',
  },
  {
    icon: GitBranch,
    color: '#00ff88',
    title: 'Coordination Under Uncertainty',
    desc: 'Agents learn to broadcast verified info, follow coordination messages, and form police-then-medical sequences.',
  },
  {
    icon: Zap,
    color: '#ffaa00',
    title: 'Conflicting Information',
    desc: 'Misinformation bursts, communication failures, and low-trust broadcasts are part of the state space.',
  },
  {
    icon: Shield,
    color: '#bb44ff',
    title: 'Trust-Weighted Conflict Resolution',
    desc: 'Commander override applies only when trust(commander) >= trust(target). Per-agent trust updates each step.',
  },
  {
    icon: Eye,
    color: '#ff6644',
    title: 'Causal Chain Reasoning',
    desc: 'Every reward is traceable: Delayed rescue -> Hospital overload -> Death -> -100 penalty. Not a black box.',
  },
];

export const CapabilityBanner = React.memo(() => {
  const [expanded, setExpanded] = useState(false);
  const mode = useStore((s) => s.mode);
  const agentTrust = useStore((s) => s.agentTrust);
  const avgTrust = Object.values(agentTrust).length > 0
    ? Math.round(Object.values(agentTrust).reduce((a, b) => a + b, 0) / Object.values(agentTrust).length)
    : null;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.8 }}
      style={{
        position: 'absolute',
        top: 76,
        left: 12,
        width: 200,
        zIndex: 45,
        pointerEvents: 'auto',
      }}
    >
      {/* Claim header */}
      <div
        onClick={() => setExpanded((e) => !e)}
        style={{
          background: 'rgba(5,10,22,0.92)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(0,212,255,0.2)',
          borderRadius: expanded ? '8px 8px 0 0' : 8,
          padding: '8px 10px',
          cursor: 'pointer',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
          <Brain size={11} color="#00d4ff" />
          <span style={{ fontSize: 8, fontWeight: 900, letterSpacing: '0.12em', color: '#00d4ff', textTransform: 'uppercase' }}>
            System Claim
          </span>
          <div style={{ marginLeft: 'auto' }}>
            {expanded ? <ChevronUp size={10} color="rgba(255,255,255,0.4)" /> : <ChevronDown size={10} color="rgba(255,255,255,0.4)" />}
          </div>
        </div>
        <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.85)', lineHeight: 1.45, fontStyle: 'italic' }}>
          "LLMs learn coordination under uncertainty via RL in a multi-agent crisis environment."
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 7, padding: '2px 5px', borderRadius: 3, background: 'rgba(0,212,255,0.12)', color: '#00d4ff', fontWeight: 700 }}>
            {mode === 'TRAINED' ? 'TRAINED POLICY' : 'BASELINE POLICY'}
          </span>
          {avgTrust !== null && (
            <span style={{ fontSize: 7, padding: '2px 5px', borderRadius: 3, background: 'rgba(187,68,255,0.12)', color: '#bb44ff', fontWeight: 700 }}>
              AVG TRUST {avgTrust}
            </span>
          )}
          <span style={{ fontSize: 7, padding: '2px 5px', borderRadius: 3, background: 'rgba(0,255,136,0.12)', color: '#00ff88', fontWeight: 700 }}>
            5 AGENTS LIVE
          </span>
        </div>
      </div>

      {/* Expandable capabilities */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{
              background: 'rgba(5,10,22,0.92)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(0,212,255,0.2)',
              borderTop: 'none',
              borderRadius: '0 0 8px 8px',
              overflow: 'hidden',
            }}
          >
            {CAPABILITIES.map(({ icon: Icon, color, title, desc }, i) => (
              <div
                key={i}
                style={{
                  padding: '7px 10px',
                  borderBottom: i < CAPABILITIES.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 2 }}>
                  <Icon size={9} color={color} />
                  <span style={{ fontSize: 8, fontWeight: 800, color, letterSpacing: '0.04em' }}>{title}</span>
                </div>
                <div style={{ fontSize: 7.5, color: 'rgba(255,255,255,0.6)', lineHeight: 1.4 }}>{desc}</div>
              </div>
            ))}
            <div style={{
              padding: '6px 10px',
              background: 'rgba(0,255,136,0.05)',
              borderTop: '1px solid rgba(0,255,136,0.12)',
              fontSize: 7, color: '#00ff88', fontWeight: 700, letterSpacing: '0.06em',
            }}>
              FRAMEWORK: earthquake / flood / blackout — same agents adapt
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
});
