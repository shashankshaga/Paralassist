

function createWheelchairSim(canvas, statusEl, opts = {}) {
  const WS_URL = opts.url || "ws://localhost:8765";
  const SPEED = 120;          // pixels per second at full confidence
  const FAILSAFE_MS = 1500;   // stop if no message arrives for this long
  const REAR_R = 34;
  const INK = "#1b2540", ACCENT = "#3b4cca", BODY = "#8b97b5", SKIN = "#d9b8a0";
  const ctx = canvas.getContext("2d");
  const state = { x: 90, wheel: 0, tilt: 0, command: "stop", confidence: 0, lastMsg: 0, ws: null, source: "idle" };

  function setCommand(cmd, conf, source) {
    state.command = cmd; state.confidence = conf; state.lastMsg = performance.now(); state.source = source;
  }

  function connect() {
    if (state.ws) state.ws.close();
    const ws = new WebSocket(WS_URL);
    state.ws = ws;
    ws.onopen = () => { if (statusEl) statusEl.textContent = "Connected to EEG bridge"; };
    ws.onmessage = e => { const m = JSON.parse(e.data); setCommand(m.command, m.confidence, "EEG"); };
    ws.onclose = ws.onerror = () => {
      state.ws = null; setCommand("stop", 0, "idle");
      if (statusEl) statusEl.textContent = "Disconnected: stopped";
    };
  }

  document.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") { e.preventDefault(); setCommand("forward", 1, "keyboard"); }
    if (e.key === "ArrowLeft") { e.preventDefault(); setCommand("back", 1, "keyboard"); }
    if (e.key === " " || e.key === "Escape") { e.preventDefault(); setCommand("stop", 0, "keyboard"); }
  });
  document.addEventListener("keyup", e => {
    if (state.source === "keyboard" && e.key.startsWith("Arrow")) setCommand("stop", 0, "keyboard");
  });

  function line(x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }

  function wheel(cx, cy, r, angle, spokes) {
    ctx.strokeStyle = INK; ctx.lineWidth = r > 20 ? 5 : 4;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    if (r > 20) {                                     // handrim on the big wheel
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, r - 7, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
    for (let i = 0; i < spokes; i++) {
      const a = angle + (i * Math.PI * 2) / spokes;
      line(cx, cy, cx + Math.cos(a) * (r - 3), cy + Math.sin(a) * (r - 3));
    }
    ctx.fillStyle = ACCENT;
    ctx.beginPath(); ctx.arc(cx, cy, r > 20 ? 4.5 : 3, 0, Math.PI * 2); ctx.fill();
  }

  function drawChair(ground) {
    ctx.fillStyle = "rgba(27,37,64,0.18)";            // shadow
    ctx.beginPath(); ctx.ellipse(state.x + 26, ground + 5, 66, 8, 0, 0, Math.PI * 2); ctx.fill();

    ctx.save();
    ctx.translate(state.x, ground);
    ctx.rotate(state.tilt);
    ctx.lineCap = "round"; ctx.lineJoin = "round";

    ctx.strokeStyle = BODY;                           // person
    ctx.lineWidth = 15; line(-4, -60, -14, -106);     // torso
    ctx.lineWidth = 11; line(-2, -60, 38, -62);       // thigh
    ctx.lineWidth = 9;  line(38, -62, 54, -26);       // shin
    ctx.strokeStyle = INK; ctx.lineWidth = 6; line(54, -24, 68, -24); // foot
    ctx.fillStyle = SKIN;
    ctx.beginPath(); ctx.arc(-15, -124, 11, 0, Math.PI * 2); ctx.fill(); // head
    ctx.strokeStyle = "#6f7ba0"; ctx.lineWidth = 6;   // arm
    ctx.beginPath(); ctx.moveTo(-13, -102); ctx.lineTo(8, -84); ctx.lineTo(26, -82); ctx.stroke();

    ctx.strokeStyle = ACCENT; ctx.lineWidth = 5;      // frame
    line(-20, -52, 42, -52);                          // seat
    line(-20, -52, -28, -112); line(-28, -112, -44, -114); // back rest + handle
    line(-12, -80, 32, -80); line(32, -80, 32, -52);  // armrest
    line(42, -52, 58, -22); line(58, -22, 74, -22);   // footrest
    line(-20, -52, 0, -34); line(0, -34, 62, -34);    // lower frame
    line(42, -52, 62, -34); line(62, -34, 62, -20);   // caster fork

    wheel(0, -REAR_R, REAR_R, state.wheel, 10);       // rear wheel
    wheel(62, -11, 11, state.wheel * 3, 4);           // front caster
    ctx.restore();
  }

  let last = performance.now();
  function frame(now) {
    const dt = (now - last) / 1000; last = now;
    const w = canvas.width, h = canvas.height;
    if (now - state.lastMsg > FAILSAFE_MS) state.command = "stop";   // failsafe
    const dir = state.command === "forward" ? 1 : state.command === "back" ? -1 : 0;
    const nextX = Math.max(40, Math.min(w - 40, state.x + dir * SPEED * state.confidence * dt));
    state.wheel += (nextX - state.x) / REAR_R;        // wheels turn with distance
    state.x = nextX;
    const target = -dir * state.confidence * 0.035;   // lean back when moving forward
    state.tilt += (target - state.tilt) * Math.min(1, dt * 6);

    ctx.clearRect(0, 0, w, h);                        // transparent: no background
    drawChair(h - 40);
    ctx.fillStyle = INK; ctx.font = "14px monospace";
    ctx.fillText(`${state.command.toUpperCase()}  ${(state.confidence * 100).toFixed(0)}%  (${state.source})`, 20, 26);
    ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.strokeRect(w - 150, 16, 120, 10);   // confidence bar
    ctx.fillStyle = ACCENT; ctx.fillRect(w - 150, 16, 120 * state.confidence, 10);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return { connect, state };
}


