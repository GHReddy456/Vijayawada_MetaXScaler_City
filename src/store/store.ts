import { create } from 'zustand';

// ─── Types (matches backend schema) ───────────────────────────────────────────

export type AgentRole = 'AMBULANCE' | 'FIRE_UNIT' | 'POLICE' | 'LOGISTICS' | 'COMMAND';
export type Severity   = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type CrisisType = 'FIRE' | 'UNREST' | 'MEDICAL' | 'BLOCKAGE';
export type SimStatus  = 'STOPPED' | 'PLAYING' | 'PAUSED';
export type AppMode    = 'BASELINE' | 'TRAINED';

export interface Agent {
  id: string;
  label: string;         // "A1", "A2", etc.
  role: AgentRole;
  x: number;            // 0–100 normalized %
  y: number;
  targetX?: number;
  targetY?: number;
  targetName?: string;
  state: 'IDLE' | 'ENROUTE' | 'ACTIVE' | 'BLOCKED';
  confidence: number;   // 0–1
  reliability: number;  // 0–1
  action: string;
  reasoning: string;
  /** What this agent is allowed to know (role-specific partial observability). */
  knowledge_scope?: string;
  /** Latest action line for UI / timeline. */
  last_decision?: string;
}

export interface Crisis {
  id: string;
  type: CrisisType;
  severity: Severity;
  name: string;
  x: number;            // 0–100 normalized %
  y: number;
  radius: number;       // visual radius px
}

export interface District {
  id: string;
  name: string;
  severity?: Severity;
  population?: number;
}

export interface Metrics {
  death_toll: number;
  panic_level: number;  // 0–100
  trust_score: number;  // 0–100
  rescue_success: number; // 0–100
  reward: number;
}

export interface EventLog {
  id: string;
  timestamp: string;
  message: string;
  severity: Severity;
}

export interface AgentCommunication {
  id: string;
  timestamp: string;
  sender: string;     // A1..A5
  recipient: string;  // A1..A5 | ALL
  text: string;
  type: 'broadcast' | 'request' | 'reply' | 'status' | 'route_advisory';
  zone?: string;
  confidence?: number;
}

/** Targeted, selective communication link (comms.py engine output) */
export interface CommLink {
  id: string;
  from: string;    // A1..A5
  to: string;      // A1..A5
  type: 'request' | 'advisory' | 'alert' | 'coordination' | 'override';
  reason: string;
  text: string;
  confidence: number;
  color: string;   // hex, pre-computed by backend
  effective: boolean | null;
  time_step: number;
}

export interface CommanderBrief {
  calamities: Array<{ kind: string; location: number[]; zone: string; intensity?: number }>;
  waiting_casualties: number;
  blocked_cells: number;
  unit_positions: Record<string, number[]>;
}

export interface SystemState {
  scenario: string;
  mode: AppMode;
  status: SimStatus;
  time_elapsed: number; // seconds
  baseline_metrics?: Partial<Metrics>;
  /** Full situational picture (commander / A5 only). */
  commander_brief?: CommanderBrief;
}

// ── Agent Chat ────────────────────────────────────────────────────────────────
export type ChatEventType = 'event' | 'decision' | 'message' | 'outcome';

export interface ChatEvent {
  type: ChatEventType;
  ts: string;
  step: number;
  text: string;
  agent?: string;
}

// ─── Store ────────────────────────────────────────────────────────────────────

interface Store {
  // WebSocket
  wsConnected: boolean;
  wsError: string | null;
  setWsConnected: (v: boolean) => void;
  setWsError: (e: string | null) => void;

  // Backend data (only from WS)
  agents: Agent[];
  crises: Crisis[];
  districts: District[];
  metrics: Metrics;
  events: EventLog[];
  communications: AgentCommunication[];
  commLinks: CommLink[];
  chatEvents: ChatEvent[];      // accumulating agent chat timeline (never overwritten)
  rewardCurve: number[];        // per-step reward accumulator for RL learning curve
  systemState: SystemState;

  // Real-time RL signals
  causalChain: string[];
  agentCorrectness: Record<string, boolean>;   // A1-A5 → green/red
  agentTrust: Record<string, number>;          // A1-A5 → 0-100

  // Local UI state
  mode: AppMode;
  setMode: (m: AppMode) => void;
  selectedAgentId: string | null;
  setSelectedAgent: (id: string | null) => void;

  // Bulk state update from WebSocket message
  applyWSMessage: (data: WSMessage) => void;
}

export interface WSMessage {
  districts?: District[];
  agents?: Agent[];
  crises?: Crisis[];
  resources?: unknown[];
  metrics?: Partial<Metrics>;
  event_log?: EventLog[];
  decision_feed?: EventLog[];
  communications?: AgentCommunication[];
  comm_links?: CommLink[];
  chat_events?: ChatEvent[];
  reward_curve?: number[];
  system_state?: Partial<SystemState>;
  causal_chain?: string[];
  agent_correctness?: Record<string, boolean>;
  agent_trust?: Record<string, number>;
}

const DEFAULT_METRICS: Metrics = {
  death_toll: 0,
  panic_level: 0,
  trust_score: 0,
  rescue_success: 0,
  reward: 0,
};

const DEFAULT_SYSTEM: SystemState = {
  scenario: 'Megaquake — Vijayawada',
  mode: 'TRAINED',
  status: 'PLAYING',
  time_elapsed: 0,
  baseline_metrics: undefined,
};

export const useStore = create<Store>((set, _get) => ({
  wsConnected: false,
  wsError: null,
  setWsConnected: (v) => set({ wsConnected: v }),
  setWsError: (e) => set({ wsError: e }),

  agents: [],
  crises: [],
  districts: [],
  metrics: { ...DEFAULT_METRICS },
  events: [],
  communications: [],
  commLinks: [],
  chatEvents: [],
  rewardCurve: [],
  systemState: { ...DEFAULT_SYSTEM },

  causalChain: [],
  agentCorrectness: {},
  agentTrust: {},

  mode: 'TRAINED',
  setMode: (m) => set({ mode: m }),
  selectedAgentId: null,
  setSelectedAgent: (id) => set({ selectedAgentId: id }),

  applyWSMessage: (data) =>
    set((state) => {
      const decisions = data.decision_feed ?? [];
      const newEvents = [...decisions, ...(data.event_log ?? []), ...state.events].slice(0, 80);
      const nextSystem = data.system_state
        ? { ...state.systemState, ...data.system_state }
        : state.systemState;
      const nextMode = (data.system_state?.mode as AppMode | undefined) ?? state.mode;

      // Merge baseline_metrics from system_state into systemState
      if (data.system_state?.baseline_metrics !== undefined) {
        nextSystem.baseline_metrics = data.system_state.baseline_metrics ?? undefined;
      }

      // Keep only the last 8 comm_links (fade old ones out in the UI)
      const nextCommLinks = data.comm_links != null
        ? data.comm_links.slice(-8)
        : state.commLinks;

      // ACCUMULATE chat events — never overwrite, cap at 200
      const nextChatEvents = data.chat_events && data.chat_events.length > 0
        ? [...state.chatEvents, ...data.chat_events].slice(-200)
        : state.chatEvents;

      const nextRewardCurve = data.reward_curve != null ? data.reward_curve : state.rewardCurve;

      return {
        agents:           data.agents    ?? state.agents,
        crises:           data.crises    ?? state.crises,
        districts:        data.districts ?? state.districts,
        metrics:          data.metrics   ? { ...state.metrics, ...data.metrics } : state.metrics,
        events:           newEvents,
        communications:   data.communications ?? state.communications,
        commLinks:        nextCommLinks,
        chatEvents:       nextChatEvents,
        rewardCurve:      nextRewardCurve,
        mode:             nextMode,
        systemState:      nextSystem,
        causalChain:      data.causal_chain   ?? state.causalChain,
        agentCorrectness: data.agent_correctness ?? state.agentCorrectness,
        agentTrust:       data.agent_trust       ?? state.agentTrust,
      };
    }),
}));
