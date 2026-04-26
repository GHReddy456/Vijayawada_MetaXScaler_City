import React, { useRef, useEffect, useCallback, useState } from 'react';
import { MapContainer, TileLayer, ZoomControl, useMap } from 'react-leaflet';
import { Crosshair } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { useStore, Agent, Crisis, CommLink } from '../store/store';

// ─── Geographic constants ──────────────────────────────────────────────────────
const MAP_CENTER: [number, number] = [16.5020, 80.6420];
const MAP_ZOOM = 13;
const MAP_MIN_ZOOM = 11;
const MAP_MAX_ZOOM = 17;

// Normalized (0–100) ↔ real lat/lng bounds
const BOUNDS = { north: 16.570, south: 16.458, west: 80.550, east: 80.720 };

function normToLatLng(x: number, y: number): L.LatLngExpression {
  return [
    BOUNDS.north - (y / 100) * (BOUNDS.north - BOUNDS.south),
    BOUNDS.west + (x / 100) * (BOUNDS.east - BOUNDS.west),
  ];
}

function ll(map: L.Map, lat: number, lng: number) {
  const p = map.latLngToContainerPoint([lat, lng]);
  return { x: p.x, y: p.y };
}

// ─── Vijayawada city boundary (red dashed polygon) ────────────────────────────
const CITY_BOUNDARY: [number, number][] = [
  [16.564, 80.580],
  [16.562, 80.605],
  [16.558, 80.630],
  [16.552, 80.656],
  [16.543, 80.682],
  [16.530, 80.700],
  [16.514, 80.712],
  [16.495, 80.705],
  [16.476, 80.688],
  [16.465, 80.660],
  [16.460, 80.628],
  [16.463, 80.596],
  [16.473, 80.569],
  [16.494, 80.558],
  [16.518, 80.560],
  [16.542, 80.565],
  [16.560, 80.572],
];

// ─── Named locations ───────────────────────────────────────────────────────────
const LOCATIONS = [
  { name: 'MALLESWARAM', lat: 16.534, lng: 80.596, type: 'district' },
  { name: 'BENZ CIRCLE', lat: 16.524, lng: 80.632, type: 'junction' },
  { name: 'AUTONAGAR', lat: 16.538, lng: 80.658, type: 'district' },
  { name: 'RAMAVARAPPADU', lat: 16.545, lng: 80.692, type: 'district' },
  { name: 'GUNADALA', lat: 16.513, lng: 80.662, type: 'district' },
  { name: 'Vijayawada Junction', lat: 16.518, lng: 80.612, type: 'junction' },
  { name: 'Vijayawada', lat: 16.510, lng: 80.640, type: 'city' },
  { name: 'IBRAHIMPATNAM', lat: 16.482, lng: 80.598, type: 'district' },
  { name: 'BHAVANIPURAM', lat: 16.497, lng: 80.612, type: 'district' },
  { name: 'KANURU', lat: 16.474, lng: 80.664, type: 'district' },
  { name: 'PATAMATA', lat: 16.472, lng: 80.635, type: 'district' },
  { name: 'Krishna River', lat: 16.466, lng: 80.640, type: 'river' },
];

// ─── Highway badges ────────────────────────────────────────────────────────────
const HIGHWAYS = [
  { label: '544F', lat: 16.558, lng: 80.582 },
  { label: '76', lat: 16.546, lng: 80.603 },
  { label: '65', lat: 16.506, lng: 80.573 },
  { label: '16', lat: 16.474, lng: 80.682 },
  { label: '228', lat: 16.461, lng: 80.654 },
];

// ─── Role colors ───────────────────────────────────────────────────────────────
const ROLE_COLOR: Record<string, string> = {
  AMBULANCE: '#00d4ff', // A1 Blue
  LOGISTICS: '#ffaa00', // A2 Yellow
  POLICE: '#00ff88', // A3 Green
  FIRE_UNIT: '#ff3355', // A4 Red
  COMMAND: '#bb44ff', // A5 Purple
};

const ROLE_ICON: Record<string, string> = {
  AMBULANCE: '🚑',
  FIRE_UNIT: '🔥',
  POLICE: '🚔',
  LOGISTICS: '🚚',
  COMMAND: '🛸',
};

// ─── Bridge-aware routing (lat/lng) ───────────────────────────────────────────
const RIVER_LAT = 16.470; // approximate river center
const BRIDGES_LL = [
  { lat: 16.470, lng: 80.618 }, // Prakasam Barrage
  { lat: 16.469, lng: 80.642 }, // Rajiv Gandhi
];

function bridgeWaypoints(
  map: L.Map,
  fromLat: number, fromLng: number,
  toLat: number, toLng: number,
): { x: number; y: number }[] {
  const goingNorth = fromLat < RIVER_LAT && toLat > RIVER_LAT;
  const goingSouth = fromLat > RIVER_LAT && toLat < RIVER_LAT;
  if (!goingNorth && !goingSouth) return [ll(map, toLat, toLng)];

  // Nearest bridge
  const br = BRIDGES_LL.reduce((best, b) =>
    Math.abs(b.lng - fromLng) < Math.abs(best.lng - fromLng) ? b : best
  );
  return [
    ll(map, br.lat + (goingSouth ? 0.003 : -0.003), br.lng),
    ll(map, br.lat + (goingSouth ? -0.003 : 0.003), br.lng),
    ll(map, toLat, toLng),
  ];
}

// ─── Recenter map (default Vijayawada view) ───────────────────────────────────
function MapRecenterControl() {
  const map = useMap();
  return (
    <div
      className="leaflet-bottom leaflet-right"
      style={{
        marginBottom: 14,
        marginRight: 12,
        zIndex: 1200,
        pointerEvents: 'auto',
      }}
    >
      <button
        type="button"
        title="Reset map to full city view"
        onClick={(e) => {
          e.stopPropagation();
          map.flyTo(MAP_CENTER, MAP_ZOOM, { duration: 0.45 });
        }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '8px 12px',
          borderRadius: 8,
          cursor: 'pointer',
          border: '1px solid rgba(0, 212, 255, 0.35)',
          background: 'rgba(5, 12, 28, 0.92)',
          color: '#00d4ff',
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          boxShadow: '0 4px 18px rgba(0,0,0,0.45)',
        }}
      >
        <Crosshair size={14} strokeWidth={2.2} />
        Recenter
      </button>
    </div>
  );
}

// ─── Canvas overlay ────────────────────────────────────────────────────────────
interface DispAgent { dx: number; dy: number; angle: number }

function SimulationCanvas() {
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const dashRef = useRef(0);
  const lastTime = useRef(performance.now());
  const displayRef = useRef<Map<string, DispAgent>>(new Map());

  const drawFrame = useCallback((now: number) => {
    const canvas = canvasRef.current;
    if (!canvas) { rafRef.current = requestAnimationFrame(drawFrame); return; }
    const ctx = canvas.getContext('2d');
    if (!ctx) { rafRef.current = requestAnimationFrame(drawFrame); return; }

    const W = canvas.width;
    const H = canvas.height;
    const dt = Math.min((now - lastTime.current) / 1000, 0.1);
    lastTime.current = now;
    dashRef.current -= 22 * dt;

    const { agents, crises, wsConnected, selectedAgentId, commLinks, agentTrust } = useStore.getState();
    const t = now / 1000;

    // ── Interpolate agent screen positions ──────────────────────────────────
    // Speed = 5.5 → smooth ~0.8s transition at 60fps for a 1-cell grid move.
    // Each WS tick (1s) the backend sends a new position; we interpolate to it.
    agents.forEach((agent) => {
      const latlng = normToLatLng(agent.x, agent.y);
      const pt = map.latLngToContainerPoint(latlng as L.LatLngExpression);
      const tx = pt.x, ty = pt.y;
      let d = displayRef.current.get(agent.id);
      if (!d) { d = { dx: tx, dy: ty, angle: 0 }; displayRef.current.set(agent.id, d); }
      const speed = 5.5; // faster lerp — fully reaches new pos in ~0.6s
      d.dx += (tx - d.dx) * Math.min(speed * dt, 1);
      d.dy += (ty - d.dy) * Math.min(speed * dt, 1);
      // Update heading based on movement direction
      const mvx = tx - d.dx, mvy = ty - d.dy;
      if (Math.hypot(mvx, mvy) > 0.5)
        d.angle += (Math.atan2(mvy, mvx) - d.angle) * 0.22;
    });
    displayRef.current.forEach((_, id) => {
      if (!agents.find(a => a.id === id)) displayRef.current.delete(id);
    });

    ctx.clearRect(0, 0, W, H);

    // ── City boundary (red dashed polygon) ────────────────────────────────
    const bPts = CITY_BOUNDARY.map(([lat, lng]) => {
      const p = map.latLngToContainerPoint([lat, lng]);
      return { x: p.x, y: p.y };
    });
    ctx.save();
    ctx.beginPath();
    bPts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.strokeStyle = 'rgba(255, 50, 50, 0.9)';
    ctx.lineWidth = 2.2;
    ctx.setLineDash([8, 6]);
    ctx.lineDashOffset = -dashRef.current * 0.5;
    ctx.shadowColor = 'rgba(255,50,50,0.5)';
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // ── Highway badges ─────────────────────────────────────────────────────
    HIGHWAYS.forEach(hw => {
      const p = map.latLngToContainerPoint([hw.lat, hw.lng]);
      ctx.save();
      const s = `${hw.label}`;
      ctx.font = 'bold 9px Inter, sans-serif';
      const tw = ctx.measureText(s).width;
      const bw = tw + 10, bh = 14;
      ctx.fillStyle = 'rgba(20,30,60,0.85)';
      ctx.strokeStyle = 'rgba(150,180,255,0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(p.x - bw / 2, p.y - bh / 2, bw, bh, 3);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = 'rgba(200,220,255,0.9)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(s, p.x, p.y);
      ctx.restore();
    });

    // ── Location labels ────────────────────────────────────────────────────
    LOCATIONS.forEach(loc => {
      const p = map.latLngToContainerPoint([loc.lat, loc.lng]);
      ctx.save();
      const isCity = loc.type === 'city';
      const isJunction = loc.type === 'junction';
      const isRiver = loc.type === 'river';
      const fSize = isCity ? 16 : isJunction ? 10 : isRiver ? 11 : 9;
      const fWeight = isCity ? '700' : isJunction ? '600' : '500';
      ctx.font = `${fWeight} ${fSize}px Inter, sans-serif`;

      if (isRiver) {
        ctx.fillStyle = 'rgba(100,160,255,0.6)';
        ctx.font = 'italic 600 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(loc.name, p.x, p.y);
        ctx.restore();
        return;
      }

      if (isCity) {
        ctx.fillStyle = 'rgba(220,235,255,0.85)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.shadowColor = 'rgba(100,150,255,0.4)';
        ctx.shadowBlur = 8;
        ctx.fillText(loc.name, p.x, p.y);
        ctx.restore();
        return;
      }

      // dot
      const dotR = isJunction ? 3.5 : 2.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
      ctx.fillStyle = isJunction ? 'rgba(180,210,255,0.85)' : 'rgba(120,160,220,0.6)';
      ctx.shadowColor = 'rgba(120,180,255,0.6)';
      ctx.shadowBlur = isJunction ? 6 : 3;
      ctx.fill();
      ctx.shadowBlur = 0;

      // label pill
      const tw2 = ctx.measureText(loc.name).width;
      const px = p.x - tw2 / 2 - 4, py = p.y + dotR + 3;
      ctx.fillStyle = 'rgba(5,12,28,0.72)';
      ctx.beginPath();
      ctx.roundRect(px, py, tw2 + 8, fSize + 4, 3);
      ctx.fill();
      ctx.fillStyle = isJunction ? 'rgba(200,225,255,0.95)' : 'rgba(140,175,230,0.78)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(loc.name, p.x, py + 2);
      ctx.restore();
    });

    // ── Crisis zones ───────────────────────────────────────────────────────
    crises.forEach(crisis => {
      const latlng = normToLatLng(crisis.x, crisis.y);
      const pt = map.latLngToContainerPoint(latlng as L.LatLngExpression);
      const cx = pt.x, cy = pt.y;
      const r = crisis.radius;
      const pulse = 0.82 + Math.sin(t * 2.4) * 0.18;
      const alphaP = 0.14 + Math.sin(t * 2.4) * 0.06;
      const col = crisisCol(crisis.severity);

      ctx.save();
      // Outer pulse
      ctx.beginPath();
      ctx.arc(cx, cy, r * pulse * 1.7, 0, Math.PI * 2);
      ctx.fillStyle = col.replace('X', String(alphaP * 0.35));
      ctx.fill();
      // Inner radial glow
      const rg = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      rg.addColorStop(0, col.replace('X', '0.6'));
      rg.addColorStop(0.5, col.replace('X', '0.22'));
      rg.addColorStop(1, col.replace('X', '0'));
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = rg;
      ctx.fill();
      // Center icon (fire emoji via text)
      ctx.font = '18px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(crisisIcon(crisis.type), cx, cy);
      // Label
      ctx.font = '600 9px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillText(crisis.name, cx, cy - r - 8);
      ctx.restore();
    });

    // ── Agent paths (bridge-aware) with moving dot ────────────────────────
    agents.forEach(agent => {
      if (agent.targetX == null || agent.targetY == null) return;
      const disp = displayRef.current.get(agent.id);
      if (!disp) return;

      // Don't draw route if agent is at/very near target
      const tLatLng = normToLatLng(agent.targetX, agent.targetY);
      const tPt = map.latLngToContainerPoint(tLatLng as L.LatLngExpression);
      const distToTarget = Math.hypot(tPt.x - disp.dx, tPt.y - disp.dy);
      if (distToTarget < 6) return;

      const color = ROLE_COLOR[agent.role] ?? '#ffffff';
      const fromLatLng = normToLatLng(agent.x, agent.y) as [number, number];
      const toLatLng = normToLatLng(agent.targetX, agent.targetY) as [number, number];
      const wps = bridgeWaypoints(map, fromLatLng[0], fromLatLng[1], toLatLng[0], toLatLng[1]);

      ctx.save();
      // Glow shadow
      ctx.beginPath();
      ctx.moveTo(disp.dx, disp.dy);
      wps.forEach(w => ctx.lineTo(w.x, w.y));
      ctx.strokeStyle = color + '33';
      ctx.lineWidth = 4;
      ctx.lineCap = ctx.lineJoin = 'round';
      ctx.stroke();
      // Animated dash line
      ctx.beginPath();
      ctx.moveTo(disp.dx, disp.dy);
      wps.forEach(w => ctx.lineTo(w.x, w.y));
      ctx.strokeStyle = color + 'dd';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      ctx.lineDashOffset = dashRef.current;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.stroke();
      ctx.setLineDash([]);

      // Moving dot travelling along route (t cycles 0→1 every 2s)
      const dotT = ((t * 0.5) % 1.0);
      const lastWp = wps[wps.length - 1];
      const dotX = disp.dx + (lastWp.x - disp.dx) * dotT;
      const dotY = disp.dy + (lastWp.y - disp.dy) * dotT;
      ctx.beginPath();
      ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.restore();
    });

    // ── Comm-link lines (targeted, selective communication) ───────────────
    // Each link is a colored arc from sender agent → receiver agent.
    // Color:  blue=request, green=advisory, red=alert, purple=coordination
    // Animated particle travels along the arc to show message in flight.
    if (commLinks && commLinks.length > 0) {
      commLinks.forEach((link: CommLink) => {
        const fromAgent = agents.find(a => a.id === link.from);
        const toAgent   = agents.find(a => a.id === link.to);
        if (!fromAgent || !toAgent) return;

        const fromDisp = displayRef.current.get(fromAgent.id);
        const toDisp   = displayRef.current.get(toAgent.id);
        if (!fromDisp || !toDisp) return;

        const fx = fromDisp.dx, fy = fromDisp.dy;
        const tx = toDisp.dx,   ty = toDisp.dy;
        if (Math.hypot(tx - fx, ty - fy) < 4) return;

        const color = link.color || '#4499ff';

        // Age-based alpha: new links are bright, fade over ~3 seconds
        const age = (t - link.time_step);   // rough age in sim-seconds
        const alpha = Math.max(0.15, Math.min(1.0, 1.0 - age * 0.08));

        // Control point for bezier arc (arc up/down depending on link type)
        const mx = (fx + tx) / 2;
        const my = (fy + ty) / 2;
        const dx = tx - fx, dy2 = ty - fy;
        const perp = Math.hypot(dx, dy2);
        const arcHeight = Math.min(60, perp * 0.35) * (link.type === 'alert' ? -1 : 1);
        const cpx = mx - (dy2 / perp) * arcHeight;
        const cpy = my + (dx / perp) * arcHeight;

        ctx.save();
        // Glow shadow
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.quadraticCurveTo(cpx, cpy, tx, ty);
        ctx.strokeStyle = color + Math.round(alpha * 0x33).toString(16).padStart(2, '0');
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.stroke();
        // Main line
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.quadraticCurveTo(cpx, cpy, tx, ty);
        const hexAlpha = Math.round(alpha * 0xcc).toString(16).padStart(2, '0');
        ctx.strokeStyle = color + hexAlpha;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.lineDashOffset = -t * 18;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;

        // Animated particle along bezier (t cycles 0→1 over 1.4s)
        const pt2 = (t * 0.7) % 1.0;
        const bx = (1 - pt2) * (1 - pt2) * fx + 2 * (1 - pt2) * pt2 * cpx + pt2 * pt2 * tx;
        const by2 = (1 - pt2) * (1 - pt2) * fy + 2 * (1 - pt2) * pt2 * cpy + pt2 * pt2 * ty;
        ctx.beginPath();
        ctx.arc(bx, by2, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Message type label near midpoint
        const lx = (fx + cpx + tx) / 3;
        const ly = (fy + cpy + ty) / 3;
        const typeLabel = link.type.toUpperCase().slice(0, 3);
        ctx.font = 'bold 7px Inter, sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = alpha * 0.9;
        ctx.fillText(typeLabel, lx, ly - 7);
        ctx.globalAlpha = 1.0;

        // Effectiveness indicator (✓ green or ✗ red) near receiver
        if (link.effective === true) {
          ctx.font = 'bold 11px sans-serif';
          ctx.fillStyle = '#00ff88';
          ctx.fillText('✓', tx + 14, ty - 10);
        } else if (link.effective === false) {
          ctx.font = 'bold 11px sans-serif';
          ctx.fillStyle = '#ff3355';
          ctx.fillText('✗', tx + 14, ty - 10);
        }
        ctx.restore();
      });
    }

    // ── Agents ────────────────────────────────────────────────────────────
    agents.forEach(agent => {
      const disp = displayRef.current.get(agent.id);
      if (!disp) return;
      const { dx, dy, angle } = disp;
      const color = ROLE_COLOR[agent.role] ?? '#fff';
      const label = agent.label || agent.id;
      const R = 16;

      ctx.save();
      ctx.translate(dx, dy);

      // Multi-layer outer glow
      const gg = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 3.5);
      gg.addColorStop(0, color + '55');
      gg.addColorStop(0.5, color + '22');
      gg.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(0, 0, R * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = gg;
      ctx.fill();

      // Main Circle with Gradient
      const circleGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, R);
      circleGrad.addColorStop(0, 'rgba(15, 32, 64, 0.95)');
      circleGrad.addColorStop(1, 'rgba(5, 12, 28, 1)');

      ctx.beginPath();
      ctx.arc(0, 0, R, 0, Math.PI * 2);
      ctx.fillStyle = circleGrad;
      ctx.shadowColor = color;
      ctx.shadowBlur = 18;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Role Icon
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(ROLE_ICON[agent.role] ?? '?', 0, -2);

      // Label Badge
      ctx.font = 'bold 8px Inter, sans-serif';
      ctx.fillStyle = color;
      ctx.fillText(label, 0, 7);

      ctx.restore();

      // State pulse ring (only for active/enroute)
      if (agent.state === 'ACTIVE' || agent.state === 'ENROUTE') {
        const pr = R + 6 + Math.sin(t * 4.5) * 4;
        ctx.save();
        ctx.beginPath();
        ctx.arc(dx, dy, pr, 0, Math.PI * 2);
        ctx.strokeStyle = color + '66';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 4]);
        ctx.stroke();
        ctx.restore();
      }

      if (selectedAgentId === agent.id) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(dx, dy, R + 11, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.95)';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.restore();
      }

      // ── Trust bar above agent ────────────────────────────────────────────
      const trustPct = (agentTrust?.[agent.label] ?? 50) / 100;
      const barW = 34, barH = 4;
      const barX = dx - barW / 2, barY = dy - R - 14;
      ctx.save();
      // background track
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath();
      ctx.roundRect(barX, barY, barW, barH, 2);
      ctx.fill();
      // fill
      const trustColor = trustPct > 0.65 ? '#00ff88' : trustPct > 0.35 ? '#ffaa00' : '#ff3355';
      ctx.fillStyle = trustColor;
      ctx.shadowColor = trustColor;
      ctx.shadowBlur = 5;
      ctx.beginPath();
      ctx.roundRect(barX, barY, barW * trustPct, barH, 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      // "TRUST" micro label
      ctx.font = '600 6px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`TRUST ${Math.round(trustPct * 100)}`, dx, barY);
      ctx.restore();
    });

    // ── Map legend ─────────────────────────────────────────────────────────
    drawLegend(ctx, W, H);

    // ── Offline indicator ──────────────────────────────────────────────────
    if (!wsConnected && agents.length === 0 && Math.sin(t * 2) > 0) {
      ctx.save();
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255,51,85,0.9)';
      ctx.textAlign = 'center';
      ctx.shadowColor = 'rgba(255,51,85,0.5)';
      ctx.shadowBlur = 10;
      ctx.fillText('⚡  AWAITING BACKEND  ·  /ws', W / 2, H / 2 + 24);
      ctx.restore();
    }

    rafRef.current = requestAnimationFrame(drawFrame);
  }, [map]);

  const onCanvasClick = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const clickX = ev.clientX - rect.left;
      const clickY = ev.clientY - rect.top;
      const agents = useStore.getState().agents;
      let picked: string | null = null;
      let best = 28;
      agents.forEach((agent) => {
        const latlng = normToLatLng(agent.x, agent.y);
        const pt = map.latLngToContainerPoint(latlng as L.LatLngExpression);
        const d = Math.hypot(pt.x - clickX, pt.y - clickY);
        if (d < best) {
          best = d;
          picked = agent.id;
        }
      });
      useStore.getState().setSelectedAgent(picked);
    },
    [map],
  );

  // Mount canvas + resize + rAF
  useEffect(() => {
    const container = map.getContainer();
    const canvas = canvasRef.current;
    if (!canvas || !container) return;

    const resize = () => {
      const r = container.getBoundingClientRect();
      canvas.width = r.width;
      canvas.height = r.height;
    };
    resize();
    window.addEventListener('resize', resize);
    map.on('move zoom', resize);

    lastTime.current = performance.now();
    rafRef.current = requestAnimationFrame(drawFrame);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
      map.off('move zoom', resize);
    };
  }, [map, drawFrame]);

  return (
    <canvas
      ref={canvasRef}
      onClick={onCanvasClick}
      style={{
        position: 'absolute', inset: 0,
        zIndex: 500, pointerEvents: 'auto',
        width: '100%', height: '100%',
        cursor: 'pointer',
      }}
    />
  );
}

// ─── Floating comm-link label overlay ────────────────────────────────────────
// Shows "💬 MEDIC → POLICE: Request route" near top-right of screen,
// fades out after 2.5 s, max 5 visible at once.

const MSG_COLORS: Record<string, string> = {
  request:      '#4499ff',
  advisory:     '#00ff88',
  alert:        '#ff3355',
  coordination: '#bb44ff',
  override:     '#ff9900',
};

const AGENT_SHORT: Record<string, string> = {
  A1: '🚑 MEDIC',
  A2: '🚚 LOGI',
  A3: '🚔 POLICE',
  A4: '📡 COMMS',
  A5: '🛸 CMD',
};

interface FloatingLabel { id: string; text: string; color: string; born: number }

function FloatingCommLabels() {
  const commLinks = useStore((s) => s.commLinks);
  const [labels, setLabels] = useState<FloatingLabel[]>([]);
  const seenRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!commLinks.length) return;
    const now = Date.now();
    const newLabels: FloatingLabel[] = [];
    commLinks.forEach((cl) => {
      if (seenRef.current.has(cl.id)) return;
      seenRef.current.add(cl.id);
      const fr   = AGENT_SHORT[cl.from] ?? cl.from;
      const to   = AGENT_SHORT[cl.to]   ?? cl.to;
      const type = (cl.type ?? 'request').toUpperCase();
      const body = cl.text ? cl.text.slice(0, 45) : type;
      newLabels.push({
        id: cl.id,
        text: `💬 ${fr} → ${to}: ${body}`,
        color: MSG_COLORS[cl.type] ?? '#4499ff',
        born: now,
      });
    });
    if (!newLabels.length) return;
    setLabels((prev) => [...prev, ...newLabels].slice(-5));
  }, [commLinks]);

  // Garbage-collect expired labels every 500 ms
  useEffect(() => {
    const timer = setInterval(() => {
      const cutoff = Date.now() - 2800;
      setLabels((prev) => prev.filter((l) => l.born > cutoff));
    }, 500);
    return () => clearInterval(timer);
  }, []);

  if (!labels.length) return null;

  return (
    <div style={{
      position: 'absolute',
      top: 76,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 600,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 4,
      pointerEvents: 'none',
    }}>
      {labels.map((lbl) => {
        const age    = Date.now() - lbl.born;
        const fading = age > 1800;
        return (
          <div
            key={lbl.id}
            style={{
              padding: '4px 12px',
              borderRadius: 20,
              background: `${lbl.color}18`,
              border: `1px solid ${lbl.color}55`,
              color: lbl.color,
              fontSize: 11,
              fontWeight: 600,
              fontFamily: 'Inter, sans-serif',
              letterSpacing: '0.03em',
              whiteSpace: 'nowrap',
              backdropFilter: 'blur(8px)',
              boxShadow: `0 0 12px ${lbl.color}33`,
              opacity: fading ? 0 : 1,
              transition: fading ? 'opacity 1s ease-out' : 'opacity 0.2s ease-in',
            }}
          >
            {lbl.text}
          </div>
        );
      })}
    </div>
  );
}

// ─── Legend ───────────────────────────────────────────────────────────────────
const LEGEND_ITEMS = [
  { label: 'AMBULANCE', icon: '🚑', color: '#00d4ff' },
  { label: 'FIRE UNIT', icon: '🔥', color: '#ff3355' },
  { label: 'POLICE', icon: '🚔', color: '#00ff88' },
  { label: 'LOGISTICS', icon: '🚚', color: '#ffaa00' },
  { label: 'COMMAND', icon: '🛸', color: '#bb44ff' },
  { label: 'HOSPITAL', icon: '✚', color: '#00d4ff' },
  { label: 'SHELTER', icon: '🏠', color: '#00ff88' },
  { label: 'BLOCKED', icon: '✕', color: '#ff3355' },
];

function drawLegend(ctx: CanvasRenderingContext2D, W: number, H: number) {
  const itemW = 90, h = 26, startX = W * 0.20, y = H - 46;
  const totalW = LEGEND_ITEMS.length * itemW;
  // backdrop
  ctx.save();
  ctx.fillStyle = 'rgba(5,12,28,0.78)';
  ctx.beginPath();
  ctx.roundRect(startX - 8, y - 4, totalW + 16, h, 6);
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,100,255,0.2)';
  ctx.lineWidth = 1;
  ctx.stroke();
  LEGEND_ITEMS.forEach((item, i) => {
    const x = startX + i * itemW;
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = item.color;
    ctx.fillText(item.icon, x, y + h / 2 - 1);
    ctx.font = '700 8px Inter, sans-serif';
    ctx.fillStyle = 'rgba(180,200,230,0.85)';
    ctx.fillText(item.label, x + 16, y + h / 2);
  });
  ctx.restore();
}

// ─── Root CityMap ──────────────────────────────────────────────────────────────
export const CityMap = React.memo(() => (
  <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
    {/* Floating comm message labels above the map */}
    <FloatingCommLabels />
    <MapContainer
      center={MAP_CENTER}
      zoom={MAP_ZOOM}
      minZoom={MAP_MIN_ZOOM}
      maxZoom={MAP_MAX_ZOOM}
      style={{ width: '100%', height: '100%' }}
      zoomControl={false}
      attributionControl={false}
      scrollWheelZoom={true}
      dragging={true}
      doubleClickZoom={true}
      touchZoom={true}
      boxZoom={true}
    >
      <ZoomControl position="bottomleft" />
      {/* Dark CartoDB tiles */}
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        subdomains="abcd"
        maxZoom={20}
      />
      {/* Simulation layer on top */}
      <SimulationCanvas />
      <MapRecenterControl />
    </MapContainer>
  </div>
));

// ─── Helpers ──────────────────────────────────────────────────────────────────
function crisisCol(severity: string): string {
  switch (severity) {
    case 'CRITICAL': return 'rgba(187,68,255,X)';
    case 'HIGH': return 'rgba(255,51,68,X)';
    case 'MEDIUM': return 'rgba(255,160,0,X)';
    default: return 'rgba(0,220,130,X)';
  }
}
function crisisIcon(type: string): string {
  switch (type) {
    case 'FIRE': return '🔥';
    case 'UNREST': return '⚠️';
    case 'MEDICAL': return '🏥';
    case 'BLOCKAGE': return '🚧';
    default: return '❗';
  }
}
