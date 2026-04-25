import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore, Agent } from '../store/store';
import { Info, CheckCircle, XCircle, Shield } from 'lucide-react';

const ROLE_COLOR: Record<string, string> = {
  AMBULANCE: '#00d4ff',
  FIRE_UNIT: '#ff4422',
  POLICE:    '#4488ff',
  LOGISTICS: '#ffaa00',
  COMMAND:   '#bb44ff',
};

function ProgressBar({ value, color, label }: { value: number; color: string; label: string }) {
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ fontSize: 7, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.05em' }}>{label}</span>
        <span style={{ fontSize: 8, color, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{Math.round(value * 100)}%</span>
      </div>
      <div style={{ height: 2, background: 'rgba(255,255,255,0.1)', borderRadius: 1, overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          style={{ height: '100%', background: color, boxShadow: `0 0 4px ${color}` }}
        />
      </div>
    </div>
  );
}

function AgentCard({
  agent, selected, onSelect, isCorrect, trustScore,
}: {
  agent: Agent; selected: boolean; onSelect: () => void;
  isCorrect?: boolean; trustScore?: number;
}) {
  const color = ROLE_COLOR[agent.role] ?? '#aaa';
  const correctColor = isCorrect === true ? '#00ff88' : isCorrect === false ? '#ff3355' : undefined;

  return (
    <div style={{
      background: 'rgba(11, 18, 32, 0.6)',
      border: selected
        ? `1px solid ${color}`
        : correctColor
          ? `1px solid ${correctColor}55`
          : `1px solid rgba(0, 180, 255, 0.15)`,
      borderRadius: 8,
      padding: '10px',
      marginBottom: 8,
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      cursor: 'pointer',
      boxShadow: selected ? `0 0 12px ${color}55` : 'none',
      transition: 'all 0.2s ease',
    }}>
      <button
        onClick={onSelect}
        style={{ all: 'unset', display: 'block' }}
      >
      {/* Top Section */}
      <div style={{ display: 'flex', gap: 10 }}>
        {/* Badge with correctness indicator */}
        <div style={{ position: 'relative', flexShrink: 0 }}>
          <div style={{
            width: 24, height: 24, borderRadius: 4,
            background: color, color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, fontWeight: 800,
            boxShadow: `0 0 10px ${color}55`,
          }}>
            {agent.label || agent.id}
          </div>
          {isCorrect !== undefined && (
            <div style={{ position: 'absolute', bottom: -4, right: -4 }}>
              {isCorrect
                ? <CheckCircle size={10} color="#00ff88" fill="#001a0a" />
                : <XCircle size={10} color="#ff3355" fill="#1a0008" />}
            </div>
          )}
        </div>
        
        {/* Details Grid */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
          {/* Col 1: Identity */}
          <div>
            <div style={{ fontSize: 8, fontWeight: 700, color: '#fff', letterSpacing: '0.05em', textTransform: 'uppercase' }}>{agent.role.replace('_', ' ')}</div>
            <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', marginTop: 1 }}>{agent.action || agent.state}</div>
          </div>
          
          {/* Col 2: Target */}
          <div>
            <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.05em' }}>TARGET</div>
            <div style={{ fontSize: 8, color: '#9bb8e8', fontWeight: 600, marginTop: 1 }}>{agent.targetName || '---'}</div>
          </div>
          
          {/* Col 3: Why */}
          <div>
            <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.05em' }}>WHY THIS ACTION?</div>
            <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.6)', lineHeight: 1.2, marginTop: 1 }}>
              {agent.reasoning?.split('. ')[0] || '---'}
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bars */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <ProgressBar label="CONFIDENCE" value={agent.confidence} color={color} />
        <ProgressBar label="RELIABILITY" value={agent.reliability} color={color} />
      </div>

      <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.06em', marginBottom: 3 }}>KNOWLEDGE SCOPE</div>
        <div style={{ fontSize: 8, color: 'rgba(200,220,255,0.85)', lineHeight: 1.35 }}>
          {agent.knowledge_scope ?? 'Operational view from backend observation.'}
        </div>
        <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.06em', marginTop: 6, marginBottom: 3 }}>LAST DECISION</div>
        <div style={{ fontSize: 8, color: '#00ffaa', fontFamily: 'var(--font-mono)', lineHeight: 1.3 }}>
          {agent.last_decision ?? `${agent.action} → ${agent.targetName ?? '—'}`}
        </div>
        {trustScore !== undefined && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 4 }}>
            <Shield size={8} color="#bb44ff" />
            <span style={{ fontSize: 7, color: 'rgba(187,68,255,0.8)' }}>TRUST</span>
            <div style={{ flex: 1, height: 2, background: 'rgba(255,255,255,0.08)', borderRadius: 1, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${trustScore}%`, background: '#bb44ff' }} />
            </div>
            <span style={{ fontSize: 8, color: '#bb44ff', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{Math.round(trustScore)}</span>
          </div>
        )}
      </div>
      </button>
    </div>
  );
}


export const AgentPanel = React.memo(() => {
  const agents = useStore((s) => s.agents);
  const selectedAgentId = useStore((s) => s.selectedAgentId);
  const setSelectedAgent = useStore((s) => s.setSelectedAgent);
  const commanderBrief = useStore((s) => s.systemState.commander_brief);
  const agentCorrectness = useStore((s) => s.agentCorrectness);
  const agentTrust = useStore((s) => s.agentTrust);
  const causalChain = useStore((s) => s.causalChain);
  const selected = selectedAgentId ? agents.find((a) => a.id === selectedAgentId) ?? null : null;

  return (
    <div
      className="glass"
      style={{
        position: 'absolute',
        top: 76,
        right: 12,
        width: 320,
        maxHeight: 'calc(100vh - 180px)',
        overflowY: 'auto',
        zIndex: 40,
        padding: '12px',
        pointerEvents: 'auto',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <Info size={12} color="#00d4ff" />
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.8)' }}>
          Agent Intelligence
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 9, color: 'rgba(255,255,255,0.4)' }}>
          {agents.length} units
        </span>
      </div>

      <AnimatePresence>
        {agents.length === 0 ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ textAlign: 'center', padding: '20px 0', color: 'rgba(255,255,255,0.3)', fontSize: 11 }}
          >
            Awaiting agent data…
          </motion.div>
        ) : (
          agents.map((agent) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              selected={selectedAgentId === agent.id}
              onSelect={() => setSelectedAgent(agent.id)}
              isCorrect={agentCorrectness[agent.label] !== undefined
                ? agentCorrectness[agent.label]
                : undefined}
              trustScore={agentTrust[agent.label]}
            />
          ))
        )}
      </AnimatePresence>

      {selected?.id === 'A5' && commanderBrief && (
        <div style={{
          marginBottom: 8, padding: '10px', borderRadius: 8,
          border: '1px solid rgba(187,68,255,0.35)', background: 'rgba(40,20,60,0.45)',
        }}>
          <div style={{ fontSize: 9, fontWeight: 800, color: '#ddaaff', letterSpacing: '0.08em', marginBottom: 6 }}>
            COMMANDER — FULL SITUATION
          </div>
          <div style={{ fontSize: 8, color: 'rgba(255,255,255,0.55)', marginBottom: 4 }}>
            Calamities: {commanderBrief.calamities?.length ?? 0} · Waiting casualties: {commanderBrief.waiting_casualties} · Blocked cells: {commanderBrief.blocked_cells}
          </div>
          {(commanderBrief.calamities ?? []).map((c, i) => (
            <div key={i} style={{ fontSize: 8, color: 'rgba(255,255,255,0.85)', marginBottom: 2 }}>
              {c.kind} @ {c.zone} (grid {c.location?.join?.(',')})
            </div>
          ))}
          <div style={{ fontSize: 7, color: 'rgba(255,255,255,0.45)', marginTop: 6 }}>Unit positions (grid)</div>
          <pre style={{ fontSize: 7, color: '#aab8e8', margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'var(--font-mono)' }}>
            {JSON.stringify(commanderBrief.unit_positions ?? {}, null, 1)}
          </pre>
        </div>
      )}

      {/* Causal Chain */}
      {causalChain.length > 0 && (
        <div style={{
          marginBottom: 8, padding: '8px', borderRadius: 6,
          background: 'rgba(255,170,0,0.05)', border: '1px solid rgba(255,170,0,0.2)',
        }}>
          <div style={{ fontSize: 9, fontWeight: 800, color: '#ffaa00', letterSpacing: '0.08em', marginBottom: 6 }}>
            CAUSAL CHAIN — LAST STEP
          </div>
          {causalChain.map((line, i) => (
            <div key={i} style={{ fontSize: 8, color: 'rgba(255,255,255,0.75)', lineHeight: 1.4, marginBottom: 3, paddingLeft: 8, borderLeft: '2px solid rgba(255,170,0,0.3)' }}>
              {line}
            </div>
          ))}
        </div>
      )}

      <div style={{
        marginTop: 4, padding: '8px', borderRadius: 6, textAlign: 'center',
        background: 'rgba(0, 180, 255, 0.05)', border: '1px solid rgba(0, 180, 255, 0.15)',
        fontSize: 9, color: 'rgba(0, 212, 255, 0.8)', fontWeight: 600,
        letterSpacing: '0.05em', cursor: 'pointer'
      }}>
        CLICK AGENT ON MAP TO INSPECT DECISION <Info size={10} style={{ verticalAlign: 'middle', marginLeft: 4 }} />
      </div>
    </div>
  );
});
