const API = location.origin;
const WS = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live";
let ws = null, reconnect = null;

function nav(page) {
  document.querySelectorAll(".nav").forEach(n => n.classList.toggle("active", n.dataset.page === page));
  document.querySelectorAll(".page").forEach(p => p.classList.remove("on"));
  const el = document.getElementById("page-" + page);
  if (el) el.classList.add("on");
  if (page === "settings") loadAuthList();
  if (page === "tracking") drawMap(window._tracks || [], "trackCanvasBig");
}
document.querySelectorAll(".nav").forEach(n => n.addEventListener("click", e => { e.preventDefault(); nav(n.dataset.page); }));

function tickClock() {
  const n = new Date();
  setT("headerTime", n.toLocaleTimeString("en-GB", { hour12: false }));
  setT("headerDate", n.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }));
  setT("drDate", n.toLocaleDateString("en-GB"));
}
setInterval(tickClock, 1000); tickClock();

function connectWS() {
  if (ws && (ws.readyState === 0 || ws.readyState === 1)) return;
  ws = new WebSocket(WS);
  ws.onopen = () => { setT("wsState", "connected", "ok"); setT("setWs", "connected"); };
  ws.onclose = () => { setT("wsState", "disconnected", "bad"); setT("setWs", "disconnected"); if (!reconnect) reconnect = setTimeout(() => { reconnect = null; connectWS(); }, 2000); };
  ws.onmessage = (e) => { try { apply(JSON.parse(e.data)); } catch (_) {} };
}
async function poll() {
  try {
    const r = await fetch(API + "/api/status");
    if (r.ok) { setT("apiState", "ok", "ok"); setT("setApi", "ok"); const j = await r.json(); if (j.data) sys(j.data); }
    else setT("apiState", "err", "bad");
  } catch { setT("apiState", "down", "bad"); }
}

function apply(s) {
  if (!s) return;
  setT("lastUpdate", "Last update: " + new Date().toLocaleTimeString());
  setT("setLast", new Date().toLocaleTimeString());
  if (s.system) sys(s.system);
  if (s.tracks) tracks(s.tracks);
  if (s.rf) rf(s.rf);
  if (s.alerts) alerts(s.alerts);
  if (s.events) events(s.events);
}

function sys(x) {
  const st = (x.status || "").toUpperCase();
  setT("sysStatusText", st === "RUNNING" ? "ACTIVE" : st || "ACTIVE");
  setT("siDevice", x.device_name || x.device || "—");
  setT("siCamFps", num(x.camera_fps, 1));
  setT("siInfFps", num(x.inference_fps, 1));
  setT("siTracks", x.tracks_count ?? 0);
  setT("siCls", x.classifier_status || "—");
  setT("siRf", x.rf_status || "—");
  setT("siVram", x.vram_mb != null ? x.vram_mb + " MB" : "—");
  setT("siUptime", up(x.uptime_s));
  setT("liveFps", "FPS: " + num(x.camera_fps, 1));
  setT("liveFps2", "FPS: " + num(x.camera_fps, 1));
  setT("pgSyStatus", st); setT("pgSyUptime", up(x.uptime_s));
  setT("pgSyDevice", x.device_name || x.device || "—");
  setT("pgSyCam", num(x.camera_fps, 1)); setT("pgSyInf", num(x.inference_fps, 1));
  setT("pgSyVram", x.vram_mb != null ? x.vram_mb + " MB" : "—");
  setT("pgSyCls", x.classifier_status || "—"); setT("pgSyRf", x.rf_status || "—");
}

function tracks(list) {
  window._tracks = list || [];
  if (!list || !list.length) {
    clearPrimary();
    summary([]);
    drawMap([]);
    drawMap([], "trackCanvasBig");
    setH("pgDetTable", '<tr><td colspan="6" class="empty">No detections</td></tr>');
    setH("pgTrTable", '<tr><td colspan="6" class="empty">No tracks</td></tr>');
    return;
  }
  const order = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  const t = [...list].sort((a, b) => (order[a.threat_level] ?? 4) - (order[b.threat_level] ?? 4))[0];
  primary(t);
  summary(list);
  drawMap(list);
  drawMap(list, "trackCanvasBig");

  setH("pgDetTable", list.map(x => {
    const type = (x.classification || {}).class_name || "—";
    const auth = x.auth_status || (x.authorization || {}).status || "—";
    const th = x.threat_level || "LOW";
    return `<tr><td>${esc(x.track_id)}</td><td>${esc(x.class_name||"—")}</td><td>${num(x.confidence,2)}</td>
      <td>${esc(type)}</td><td><span class="badge ${auth}">${auth}</span></td>
      <td><span class="badge ${th}">${th}</span></td></tr>`;
  }).join(""));

  setH("pgTrTable", list.map(x => {
    const c = x.centroid, auth = x.auth_status || (x.authorization || {}).status || "—", th = x.threat_level || "LOW";
    return `<tr><td>${esc(x.track_id)}</td><td>${esc(x.status||"—")}</td><td>${x.hits??"—"}</td>
      <td class="mono">${c?num(c[0],0)+", "+num(c[1],0):"—"}</td>
      <td><span class="badge ${auth}">${auth}</span></td>
      <td><span class="badge ${th}">${th}</span></td></tr>`;
  }).join(""));
}

function primary(t) {
  const cls = t.classification || {};
  const auth = t.authorization || {};
  const authSt = t.auth_status || auth.status || "—";
  const threat = t.threat_level || "LOW";
  const v = t.velocity || [0, 0];
  const c = t.centroid;
  const speed = Math.hypot(v[0] || 0, v[1] || 0);
  const head = heading(v[0], v[1]);
  const type = cls.class_name || t.class_name || "—";
  const range = t.estimated_range_m != null ? `~${Math.round(t.estimated_range_m)} m` : "N/A";

  setT("drObject", String(t.class_name || type || "—").toUpperCase());
  setT("drConf", t.confidence != null ? (t.confidence * 100).toFixed(0) + " %" : "—");
  setT("drTime", new Date().toLocaleTimeString("en-GB", { hour12: false }));

  // Classification: show Unauthorized when UNKNOWN drone
  let classLabel = type;
  if ((type.toLowerCase().includes("drone") || (t.class_name || "").toLowerCase().includes("drone")) && authSt === "UNKNOWN") {
    classLabel = "Unauthorized";
  } else if (authSt === "AUTHORIZED") classLabel = "Authorized";
  setT("clClass", classLabel);
  setT("clConf", cls.confidence != null ? (cls.confidence * 100).toFixed(1) + " %" : (t.confidence != null ? (t.confidence * 100).toFixed(1) + " %" : "—"));
  const stEl = document.getElementById("clStatus");
  if (stEl) {
    if (threat === "HIGH" || threat === "CRITICAL") stEl.innerHTML = '<span class="badge HIGH">THREAT</span>';
    else stEl.textContent = authSt;
  }

  setT("trId", "#" + t.track_id);
  setT("trStatus", t.status || "Tracking");
  setT("trPos", c ? `${num(c[0],0)}, ${num(c[1],0)} px` : "—");
  setT("trSpeed", num(speed, 1) + " px/f");
  setT("trDir", head != null ? num(head, 0) + "°" : "—");
  setT("trRange", range);
  setT("trHits", (t.hits ?? "—") + " / " + (t.age ?? "—"));

  setT("pgDetObject", String(t.class_name || "—").toUpperCase());
  setT("pgDetConf", t.confidence != null ? (t.confidence * 100).toFixed(0) + " %" : "—");
  setT("pgDetTrack", "#" + t.track_id);
  setT("pgDetBbox", t.bbox ? t.bbox.map(n => Math.round(n)).join(", ") : "—");
  setT("pgTrId", "#" + t.track_id); setT("pgTrStatus", t.status || "—");
  setT("pgTrHits", t.hits ?? "—"); setT("pgTrAge", t.age ?? "—");
  setT("pgTrVel", num(v[0],1) + ", " + num(v[1],1)); setT("pgTrRange", range);
}

function clearPrimary() {
  ["drObject","drConf","drTime","clClass","clConf","clStatus","trId","trStatus","trPos","trSpeed","trDir","trRange","trHits",
   "pgDetObject","pgDetConf","pgDetTrack","pgDetBbox","pgTrId","pgTrStatus","pgTrHits","pgTrAge","pgTrVel","pgTrRange"
  ].forEach(id => setT(id, "—"));
  const b = document.getElementById("alertBanner");
  if (b) { b.classList.add("ok"); setT("alertTitle", "NO ACTIVE THREAT"); setT("alertSub", "System monitoring"); }
  setT("alLevel", "—"); setT("alAction", "None"); setT("alTime", "—");
}

function summary(list) {
  let d = 0, b = 0, a = 0, u = 0;
  list.forEach(t => {
    const ty = ((t.classification || {}).class_name || t.class_name || "").toLowerCase();
    if (ty.includes("drone")) d++;
    else if (ty.includes("bird")) b++;
    else if (ty.includes("aircraft") || ty.includes("airplane") || ty.includes("heli")) a++;
    else u++;
  });
  const tot = list.length;
  setT("sumTotal", tot); setT("sumDrone", d); setT("sumBird", b); setT("sumAir", a); setT("sumUnk", u);
  const donut = document.getElementById("donut");
  if (!donut) return;
  if (!tot) { donut.style.background = "conic-gradient(#334155 0 360deg)"; return; }
  const p = n => (n / tot) * 360;
  let ang = 0; const parts = [];
  [["#ef4444", d], ["#22c55e", b], ["#3b82f6", a], ["#94a3b8", u]].forEach(([c, n]) => {
    if (!n) return; parts.push(`${c} ${ang}deg ${ang + p(n)}deg`); ang += p(n);
  });
  donut.style.background = `conic-gradient(${parts.join(",")})`;
}

function rf(x) {
  const st = x.status || "RF SENSOR OFFLINE";
  setT("rfStatus", st); setT("pgRfStatus", st);
  if (st === "ok") {
    setT("rfDet", x.detected ? "Yes" : "No"); setT("pgRfDet", x.detected ? "Yes" : "No");
    const f = x.frequency_mhz != null ? x.frequency_mhz + " MHz" : (x.frequency_hz != null ? (x.frequency_hz / 1e6).toFixed(1) + " MHz" : "—");
    setT("rfFreq", f); setT("pgRfFreq", f);
    setT("rfPow", x.signal_strength_dbm != null ? x.signal_strength_dbm + " dBm" : "—");
    setT("pgRfPow", x.signal_strength_dbm != null ? x.signal_strength_dbm + " dBm" : "—");
    if (x.spectrum) { spec(x.spectrum, "spectrumCanvas"); spec(x.spectrum, "spectrumCanvas2"); }
  } else {
    ["rfDet","rfFreq","rfPow","pgRfDet","pgRfFreq","pgRfPow"].forEach(id => setT(id, "—"));
  }
}

function alerts(list) {
  const ban = document.getElementById("alertBanner");
  const pban = document.getElementById("pgAlertBanner");
  if (list && list.length) {
    const a = list[0];
    if (ban) ban.classList.remove("ok");
    if (pban) pban.classList.remove("ok");
    setT("alertTitle", "UNAUTHORIZED DRONE DETECTED");
    setT("alertSub", (a.reasons || []).slice(0, 2).join(" · ") || ("Track #" + a.track_id));
    setT("pgAlertTitle", (a.level || "HIGH") + " — TRACK #" + a.track_id);
    setT("pgAlertSub", (a.reasons || []).join(" · ") || "Alert");
    const lv = document.getElementById("alLevel");
    if (lv) lv.innerHTML = `<span class="badge ${a.level}">${a.level}</span>`;
    setT("alAction", "Alert Triggered");
    setT("alTime", a.timestamp ? new Date(a.timestamp).toLocaleTimeString("en-GB", { hour12: false }) : "—");
    setH("pgAlTable", list.map(x => {
      const tm = x.timestamp ? new Date(x.timestamp).toLocaleTimeString("en-GB", { hour12: false }) : "—";
      return `<tr><td>${tm}</td><td>#${esc(x.track_id)}</td><td><span class="badge ${x.level}">${x.level}</span></td><td>${esc((x.reasons||[]).join("; "))}</td></tr>`;
    }).join(""));
  } else {
    if (ban) ban.classList.add("ok");
    if (pban) pban.classList.add("ok");
    setT("alertTitle", "NO ACTIVE THREAT"); setT("alertSub", "System monitoring");
    setT("pgAlertTitle", "NO ACTIVE THREAT"); setT("pgAlertSub", "—");
    setT("alLevel", "—"); setT("alAction", "None"); setT("alTime", "—");
    setH("pgAlTable", '<tr><td colspan="4" class="empty">No alerts</td></tr>');
  }
}

function events(list) {
  if (!list || !list.length) { setH("pgEvTable", '<tr><td colspan="3" class="empty">No events</td></tr>'); return; }
  setH("pgEvTable", [...list].reverse().slice(0, 80).map(e => {
    const tm = e.timestamp ? new Date(e.timestamp).toLocaleTimeString("en-GB", { hour12: false }) : "—";
    return `<tr><td>${tm}</td><td>${esc(e.type||"—")}</td><td>${esc(e.message||"")}</td></tr>`;
  }).join(""));
}

async function loadAuthList() {
  try {
    const r = await fetch(API + "/api/authorization");
    if (!r.ok) return;
    const rows = (await r.json()).data || [];
    if (!rows.length) { setH("pgAuTable", '<tr><td colspan="4" class="empty">No registered drones</td></tr>'); return; }
    setH("pgAuTable", rows.map(d =>
      `<tr><td>${esc(d.drone_id)}</td><td>${esc(d.model||"—")}</td>
       <td><span class="badge AUTHORIZED">${d.status||"AUTHORIZED"}</span></td>
       <td><button class="btn-danger" onclick="deleteDrone('${esc(d.drone_id)}')">Del</button></td></tr>`
    ).join(""));
  } catch {}
}
async function registerDrone() {
  const body = {
    drone_id: document.getElementById("regId").value.trim(),
    model: document.getElementById("regModel").value.trim(),
    type: document.getElementById("regType").value.trim() || "drone",
    status: "AUTHORIZED"
  };
  if (!body.drone_id || !body.model) { setT("regMsg", "ID + Model required"); return; }
  try {
    const r = await fetch(API + "/api/authorization/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    setT("regMsg", r.ok ? "Registered ✓" : "Failed");
    if (r.ok) loadAuthList();
  } catch { setT("regMsg", "Error"); }
}
async function deleteDrone(id) {
  if (!confirm("Remove " + id + "?")) return;
  await fetch(API + "/api/authorization/" + encodeURIComponent(id), { method: "DELETE" });
  loadAuthList();
}

function frames() {
  const url = API + "/api/frame.jpg?t=" + Date.now();
  document.querySelectorAll(".liveFrame").forEach(img => {
    const probe = new Image();
    probe.onload = () => {
      img.src = url; img.style.display = "block"; img.dataset.ok = "1";
      const ph = img.parentElement && img.parentElement.querySelector(".live-ph");
      if (ph) ph.style.display = "none";
    };
    probe.onerror = () => {
      if (!img.dataset.ok) {
        img.style.display = "none";
        const ph = img.parentElement && img.parentElement.querySelector(".live-ph");
        if (ph) ph.style.display = "flex";
      }
    };
    probe.src = url;
  });
}

function spec(data, id) {
  const c = document.getElementById(id); if (!c || !data || !data.length) return;
  const ctx = c.getContext("2d"), dpr = devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight; if (!w || !h) return;
  c.width = w * dpr; c.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const mn = Math.min(...data), mx = Math.max(...data), rg = mx - mn || 1;
  ctx.beginPath(); ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 1.5;
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w, y = h - ((v - mn) / rg) * (h - 4) - 2;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function drawMap(list, id) {
  const c = document.getElementById(id || "trackCanvas"); if (!c) return;
  const ctx = c.getContext("2d"), dpr = devicePixelRatio || 1;
  const w = c.clientWidth || 320, h = c.clientHeight || 180;
  c.width = Math.max(1, w * dpr); c.height = Math.max(1, h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0a1220"; ctx.fillRect(0, 0, w, h);
  // subtle map-like grid
  ctx.strokeStyle = "#152238";
  for (let x = 0; x < w; x += 36) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y < h; y += 36) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if (!list || !list.length) {
    ctx.fillStyle = "#64748b"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("No track path", w / 2, h / 2); return;
  }
  const pts = [];
  list.forEach(t => { (t.trajectory || []).forEach(p => pts.push(p)); if (t.centroid) pts.push(t.centroid); });
  if (!pts.length) return;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  let minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  if (maxX - minX < 50) { const m = (minX + maxX) / 2; minX = m - 25; maxX = m + 25; }
  if (maxY - minY < 50) { const m = (minY + maxY) / 2; minY = m - 25; maxY = m + 25; }
  const pad = 20, sc = Math.min((w - pad * 2) / (maxX - minX), (h - pad * 2) / (maxY - minY));
  const map = p => [pad + (p[0] - minX) * sc, pad + (p[1] - minY) * sc];
  const cols = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7"];
  list.forEach((t, i) => {
    const col = cols[i % cols.length], traj = t.trajectory || [];
    if (traj.length >= 2) {
      ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.setLineDash([4, 4]);
      traj.forEach((p, j) => { const [x, y] = map(p); j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
      // start
      const [sx, sy] = map(traj[0]);
      ctx.beginPath(); ctx.fillStyle = "#f59e0b"; ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
    }
    const cur = t.centroid || (traj.length ? traj[traj.length - 1] : null);
    if (cur) {
      const [x, y] = map(cur);
      ctx.beginPath(); ctx.fillStyle = "#22c55e"; ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#e2e8f0"; ctx.font = "10px monospace"; ctx.fillText("#" + t.track_id, x + 8, y - 4);
    }
  });
}

function heading(vx, vy) {
  if (vx == null || vy == null || (Math.abs(vx) < 0.3 && Math.abs(vy) < 0.3)) return null;
  let d = Math.atan2(vx, -vy) * 180 / Math.PI; if (d < 0) d += 360; return d;
}
function setT(id, t, cls) {
  const el = document.getElementById(id); if (!el) return;
  el.textContent = t;
  if (cls !== undefined) { el.classList.remove("ok", "bad"); if (cls) el.classList.add(cls); }
}
function setH(id, h) { const el = document.getElementById(id); if (el) el.innerHTML = h; }
function num(v, d) { if (v == null || isNaN(v)) return "—"; return Number(v).toFixed(d); }
function up(s) {
  if (s == null) return "—"; s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return h + "h " + m + "m " + sec + "s"; if (m) return m + "m " + sec + "s"; return sec + "s";
}
function esc(s) { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
function toggleFs() { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }

connectWS();
frames(); setInterval(frames, 250);
poll(); setInterval(poll, 4000);
window.addEventListener("resize", () => { drawMap(window._tracks || []); drawMap(window._tracks || [], "trackCanvasBig"); });
