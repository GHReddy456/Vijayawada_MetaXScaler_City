import { useEffect, useRef } from 'react';
import { useStore, Agent, Crisis, EventLog } from '../store/store';

// BOUNDS: north=16.570, south=16.458, west=80.550, east=80.720
// x=(lng-80.550)/0.170*100   y=(16.570-lat)/0.112*100
const p = (lat: number, lng: number) => ({
  x: +((lng - 80.550) / 0.170 * 100).toFixed(2),
  y: +((16.570 - lat) / 0.112 * 100).toFixed(2),
});

// ── Road waypoints per agent (real Vijayawada roads) ─────────────────────────
const ROUTES: Record<string, {x:number;y:number}[]> = {
  A1: [ // Ambulance: Autonagar → Benz Circle → VJ Junction → back
    p(16.538,80.658), p(16.533,80.648), p(16.526,80.638),
    p(16.524,80.632), p(16.521,80.622), p(16.518,80.612),
    p(16.521,80.622), p(16.524,80.632), p(16.526,80.638), p(16.533,80.648),
  ],
  A2: [ // Logistics: VJ Junction → Gunadala → Kanuru → back
    p(16.518,80.612), p(16.518,80.632), p(16.515,80.645),
    p(16.513,80.662), p(16.500,80.662), p(16.487,80.663), p(16.474,80.664),
    p(16.487,80.663), p(16.500,80.662), p(16.513,80.662), p(16.515,80.645),
  ],
  A3: [ // Police: Benz Circle → VJ Junction → Bhavanipuram → back
    p(16.524,80.632), p(16.522,80.624), p(16.520,80.618),
    p(16.518,80.612), p(16.508,80.610), p(16.497,80.609),
    p(16.508,80.610), p(16.518,80.612), p(16.520,80.618),
  ],
  A4: [ // Fire Unit: VJ Junction → Bridge → Patamata → back
    p(16.518,80.612), p(16.512,80.614), p(16.506,80.617),
    p(16.500,80.618), p(16.492,80.619), // bridge crossing
    p(16.485,80.622), p(16.479,80.630), p(16.476,80.636),
    p(16.479,80.630), p(16.485,80.622),
    p(16.492,80.619), p(16.500,80.618), p(16.506,80.617),
  ],
  A5: [ // Command: Malleswaram → Benz Circle → Autonagar → back
    p(16.534,80.596), p(16.530,80.606), p(16.527,80.616),
    p(16.524,80.632), p(16.530,80.644), p(16.535,80.652), p(16.538,80.658),
    p(16.535,80.652), p(16.530,80.644), p(16.524,80.632), p(16.527,80.616),
  ],
};

// ── Static initial crises ────────────────────────────────────────────────────
const INIT_CRISES: Crisis[] = [
  { id:'c1', type:'FIRE',     severity:'HIGH',   name:'Benz Circle Fire',   ...p(16.524,80.632), radius:45 },
  { id:'c2', type:'FIRE',     severity:'HIGH',   name:'Patamata Fire',      ...p(16.476,80.636), radius:40 },
  { id:'c3', type:'UNREST',   severity:'MEDIUM', name:'Malleswaram Unrest', ...p(16.527,80.600), radius:35 },
  { id:'c4', type:'MEDICAL',  severity:'MEDIUM', name:'Gunadala Medical',   ...p(16.513,80.662), radius:30 },
  { id:'c5', type:'BLOCKAGE', severity:'LOW',    name:'VJ Junction Block',  ...p(16.518,80.615), radius:22 },
];

// ── Static agent definitions ─────────────────────────────────────────────────
const INIT_AGENTS: Agent[] = [
  {
    id:'A1', label:'A1', role:'AMBULANCE',
    ...ROUTES.A1[0], targetX: p(16.524,80.632).x, targetY: p(16.524,80.632).y,
    targetName:'City Hospital', state:'ENROUTE',
    confidence:0.92, reliability:0.94,
    action:'Enroute to Hospital',
    reasoning:'High severity casualty + nearest available hospital + fastest safe route',
  },
  {
    id:'A2', label:'A2', role:'LOGISTICS',
    ...ROUTES.A2[0], targetX: p(16.474,80.664).x, targetY: p(16.474,80.664).y,
    targetName:'Gunadala Shelter', state:'ENROUTE',
    confidence:0.88, reliability:0.90,
    action:'Delivering Supplies',
    reasoning:'Shelter low on supplies + high population density + accessible route',
  },
  {
    id:'A3', label:'A3', role:'POLICE',
    ...ROUTES.A3[0], targetX: p(16.524,80.632).x, targetY: p(16.524,80.632).y,
    targetName:'Benz Circle', state:'ACTIVE',
    confidence:0.85, reliability:0.88,
    action:'Restoring Order',
    reasoning:'High panic level + unrest detected + prevent escalation',
  },
  {
    id:'A4', label:'A4', role:'FIRE_UNIT',
    ...ROUTES.A4[0], targetX: p(16.476,80.636).x, targetY: p(16.476,80.636).y,
    targetName:'Tadigadapa Fire', state:'ENROUTE',
    confidence:0.90, reliability:0.93,
    action:'Extinguishing Fire',
    reasoning:'Critical fire threat + building density high + containment possible',
  },
  {
    id:'A5', label:'A5', role:'COMMAND',
    ...ROUTES.A5[0], targetX: p(16.524,80.632).x, targetY: p(16.524,80.632).y,
    targetName:'Multi-District', state:'ACTIVE',
    confidence:0.95, reliability:0.96,
    action:'Coordinating Units',
    reasoning:'Multiple crises + resource optimization + max impact',
  },
];

// ── Events ───────────────────────────────────────────────────────────────────
const EVENT_POOL: Omit<EventLog,'id'>[] = [
  { timestamp:'01:24:30', message:'Major fire outbreak in Benz Circle',          severity:'HIGH'   },
  { timestamp:'01:24:20', message:'Bridge congestion on Prakasam Barrage',       severity:'HIGH'   },
  { timestamp:'01:24:10', message:'Misinformation spreading in Malleswaram',     severity:'MEDIUM' },
  { timestamp:'01:24:00', message:'Hospital capacity reached 90%',               severity:'HIGH'   },
  { timestamp:'01:23:50', message:'Shelter opened in Gunadala',                  severity:'LOW'    },
  { timestamp:'01:23:40', message:'Aftershock detected – magnitude 5.2',         severity:'MEDIUM' },
  { timestamp:'01:23:30', message:'Supplies delivered to Bhavanipuram',          severity:'LOW'    },
  { timestamp:'01:23:20', message:'A4 containment zone established at Patamata', severity:'MEDIUM' },
  { timestamp:'01:23:10', message:'Road cleared near VJ Junction',               severity:'LOW'    },
  { timestamp:'01:23:00', message:'Communication restored in Kanuru',            severity:'LOW'    },
];

// ── Agent route state (mutable refs, not store) ───────────────────────────────
interface RouteState { wpIdx: number; progress: number }
const ROUTE_STATE: Record<string, RouteState> = {
  A1:{ wpIdx:0, progress:0 },
  A2:{ wpIdx:0, progress:0 },
  A3:{ wpIdx:0, progress:0 },
  A4:{ wpIdx:0, progress:0 },
  A5:{ wpIdx:0, progress:0 },
};
// Stagger starting positions so agents aren't all at waypoint 0
const INIT_WP = { A1:0, A2:3, A3:2, A4:0, A5:4 };

// ── Hook ─────────────────────────────────────────────────────────────────────
export function useMockSimulation() {
  const rafRef     = useRef<number>(0);
  const lastTime   = useRef(performance.now());
  const elapsed    = useRef(5000); // start at 01:23:20 equivalent
  const metricsTick = useRef(0);
  const eventTick  = useRef(0);
  const inited     = useRef(false);

  useEffect(() => {
    if (inited.current) return;
    inited.current = true;

    // Stagger start positions
    const startAgents = INIT_AGENTS.map(a => {
      const startWp = INIT_WP[a.id as keyof typeof INIT_WP] ?? 0;
      ROUTE_STATE[a.id].wpIdx = startWp;
      const route = ROUTES[a.id];
      const wp = route[startWp % route.length];
      return { ...a, x: wp.x, y: wp.y };
    });

    useStore.setState({
      agents:  startAgents,
      crises:  INIT_CRISES,
      events:  EVENT_POOL.map((e, i) => ({ ...e, id: `ev${i}` })),
      metrics: { death_toll:1247, panic_level:72, trust_score:68, rescue_success:64, reward:4837 },
      systemState: {
        scenario: 'Megaquake — Vijayawada',
        mode: 'TRAINED', status: 'PLAYING',
        time_elapsed: 5000,
        baseline_metrics: { death_toll:2847, panic_level:85, rescue_success:38, reward:-1247, trust_score:40 },
      },
    });

    // ── Simulation loop ─────────────────────────────────────────────────────
    const AGENT_SPEED = 2.8; // normalized units per second

    const tick = (now: number) => {
      const dt = Math.min((now - lastTime.current) / 1000, 0.08);
      lastTime.current = now;
      elapsed.current += dt;
      metricsTick.current += dt;
      eventTick.current   += dt;

      // Move agents along road waypoints
      const agents = useStore.getState().agents.map(agent => {
        const route = ROUTES[agent.id];
        if (!route) return agent;
        const rs = ROUTE_STATE[agent.id];
        const cur = route[rs.wpIdx % route.length];
        const nxt = route[(rs.wpIdx + 1) % route.length];

        const dx = nxt.x - cur.x;
        const dy = nxt.y - cur.y;
        const dist = Math.hypot(dx, dy);

        rs.progress += (AGENT_SPEED * dt) / (dist || 1);

        if (rs.progress >= 1) {
          rs.progress = 0;
          rs.wpIdx = (rs.wpIdx + 1) % route.length;
          const nextWp = route[rs.wpIdx % route.length];
          return { ...agent, x: nextWp.x, y: nextWp.y };
        }

        return {
          ...agent,
          x: cur.x + dx * rs.progress,
          y: cur.y + dy * rs.progress,
        };
      });

      useStore.setState({ agents, systemState: {
        ...useStore.getState().systemState,
        time_elapsed: Math.floor(elapsed.current),
      }});

      // Fluctuate metrics slightly
      if (metricsTick.current > 3) {
        metricsTick.current = 0;
        const m = useStore.getState().metrics;
        useStore.setState({ metrics: {
          death_toll:     Math.max(0,  m.death_toll    + (Math.random()-0.6)*2),
          panic_level:    Math.max(0,  Math.min(100, m.panic_level  + (Math.random()-0.55)*0.5)),
          trust_score:    Math.min(100,Math.max(0,  m.trust_score   + (Math.random()-0.45)*0.3)),
          rescue_success: Math.min(100,Math.max(0,  m.rescue_success+ (Math.random()-0.45)*0.3)),
          reward:         m.reward + (Math.random()-0.3)*8,
        }});
      }

      // Add new events periodically
      if (eventTick.current > 12) {
        eventTick.current = 0;
        const NEW_EVENTS = [
          'Rescue team dispatched to Benz Circle',
          'Water supply restored in Gunadala',
          'Medical unit reached VJ Junction',
          'Fire contained near Malleswaram',
          'Evacuation complete in Bhavanipuram',
          'A1 delivered patient to City Hospital',
          'New aftershock detected – magnitude 4.1',
          'Command coordination improved response by 12%',
        ];
        const msg = NEW_EVENTS[Math.floor(Math.random() * NEW_EVENTS.length)];
        const sevs = ['LOW','MEDIUM','HIGH'] as const;
        const sev  = sevs[Math.floor(Math.random()*3)];
        const mins = String(Math.floor(elapsed.current/60)).padStart(2,'0');
        const secs = String(Math.floor(elapsed.current%60)).padStart(2,'0');
        const ev: EventLog = { id:`live_${Date.now()}`, timestamp:`${mins}:${secs}`, message:msg, severity:sev };
        useStore.setState({ events: [ev, ...useStore.getState().events].slice(0,80) });
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);
}
