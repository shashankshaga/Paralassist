// wheelchair.js: a 1D wheelchair simulation driven by EEG commands.
// Usage: const sim = createWheelchairSim(document.getElementById("wc"), statusEl);
//        sim.connect();   // live EEG via eeg_bridge.py
// Keyboard fallback: ArrowRight = forward, ArrowLeft = back (for demos without hardware).

function createWheelchairSim(canvas, statusEl, opts = {}) {
  const WS_URL = opts.url || "ws://localhost:8765";
  const SPEED = 120;          // pixels per second at full confidence
  const FAILSAFE_MS = 1500;   // stop if no message arrives for this long
  const ctx = canvas.getContext("2d");
  const state = { x: 60, command: "stop", confidence: 0, lastMsg: 0, ws: null, source: "idle" };

  function setCommand(cmd, conf, source) {
    state.command = cmd; state.confidence = conf; state.lastMsg = performance.now(); state.source = source;
  }

  function connect() {
    if (state.ws) state.ws.close();
    const ws = new WebSocket(WS_URL);
    state.ws = ws;
    ws.onopen = () => statusEl.textContent = "Connected to EEG bridge";
    ws.onmessage = e => { const m = JSON.parse(e.data); setCommand(m.command, m.confidence, "EEG"); };
    ws.onclose = ws.onerror = () => { state.ws = null; setCommand("stop", 0, "idle"); statusEl.textContent = "Disconnected: stopped"; };
  }

  document.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") setCommand("forward", 1, "keyboard");
    if (e.key === "ArrowLeft") setCommand("back", 1, "keyboard");
    if (e.key === " " || e.key === "Escape") setCommand("stop", 0, "keyboard");
  });
  document.addEventListener("keyup", e => {
    if (state.source === "keyboard" && e.key.startsWith("Arrow")) setCommand("stop", 0, "keyboard");
  });

  let last = performance.now();
  function frame(now) {
    const dt = (now - last) / 1000; last = now;
    if (now - state.lastMsg > FAILSAFE_MS) state.command = "stop";   // failsafe
    const dir = state.command === "forward" ? 1 : state.command === "back" ? -1 : 0;
    state.x = Math.max(40, Math.min(canvas.width - 40, state.x + dir * SPEED * state.confidence * dt));
    draw();
    requestAnimationFrame(frame);
  }

  function draw() {
    const w = canvas.width, h = canvas.height, css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim() || "#1b2540";
    const accent = css.getPropertyValue("--accent").trim() || "#3b4cca";
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(20, h - 40); ctx.lineTo(w - 20, h - 40); ctx.stroke();       // floor
    ctx.fillStyle = ink; ctx.font = "14px 'IBM Plex Mono', monospace";
    ctx.fillText("START", 20, h - 18); ctx.fillText("END", w - 55, h - 18);
    const x = state.x, y = h - 40;
    ctx.strokeStyle = accent; ctx.fillStyle = accent; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.arc(x, y - 22, 22, 0, Math.PI * 2); ctx.stroke();                     // wheel
    ctx.beginPath(); ctx.moveTo(x - 10, y - 30); ctx.lineTo(x - 10, y - 70); ctx.lineTo(x - 30, y - 70); ctx.stroke(); // back rest
    ctx.beginPath(); ctx.moveTo(x - 10, y - 46); ctx.lineTo(x + 28, y - 46); ctx.stroke();      // seat
    ctx.beginPath(); ctx.arc(x - 20, y - 86, 8, 0, Math.PI * 2); ctx.fill();                   // head
    const label = `${state.command.toUpperCase()}  ${(state.confidence * 100).toFixed(0)}%  (${state.source})`;
    ctx.fillStyle = ink; ctx.fillText(label, 20, 24);
  }

  requestAnimationFrame(frame);
  return { connect, state };
}

/* Add to the page:
<canvas id="wc" width="800" height="200" style="width:100%;border:1px solid var(--line);border-radius:8px"></canvas>
<p id="wc-status" aria-live="polite"></p>
<button id="wc-connect">Connect live EEG</button>
<script src="wheelchair.js"></script>
<script>
  const sim = createWheelchairSim(document.getElementById("wc"), document.getElementById("wc-status"));
  document.getElementById("wc-connect").onclick = sim.connect;
</script>
*/
