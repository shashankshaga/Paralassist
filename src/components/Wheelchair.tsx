import { useEffect, useRef, useState } from "react";

// Wheelchair simulation: just the person in the chair on a plain background, with a soft
// shadow, spinning spoked wheels, and a slight lean when accelerating.
// Receives {"command": "forward"|"back"|"stop", "confidence": 0-1} from eeg_bridge.py.
// Arrow keys work as a fallback (Space/Esc = stop).

const WS_URL = "ws://localhost:8765";
const SPEED = 120; // pixels per second at full confidence
const FAILSAFE_MS = 1500; // stop if no message arrives for this long
const W = 800;
const H = 220;
const GROUND = H - 40;
const INK = "#1b2540";
const ACCENT = "#3b4cca";
const BODY = "#8b97b5";
const SKIN = "#d9b8a0";
const REAR_R = 34;

type Command = "forward" | "back" | "stop";

function line(c: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  c.beginPath();
  c.moveTo(x1, y1);
  c.lineTo(x2, y2);
  c.stroke();
}

function wheel(c: CanvasRenderingContext2D, cx: number, cy: number, r: number, angle: number, spokes: number) {
  c.strokeStyle = INK;
  c.lineWidth = r > 20 ? 5 : 4;
  c.beginPath();
  c.arc(cx, cy, r, 0, Math.PI * 2);
  c.stroke();
  if (r > 20) {
    c.strokeStyle = ACCENT; // handrim
    c.lineWidth = 2;
    c.beginPath();
    c.arc(cx, cy, r - 7, 0, Math.PI * 2);
    c.stroke();
  }
  c.strokeStyle = INK;
  c.lineWidth = 1.2;
  for (let i = 0; i < spokes; i++) {
    const a = angle + (i * Math.PI * 2) / spokes;
    line(c, cx, cy, cx + Math.cos(a) * (r - 3), cy + Math.sin(a) * (r - 3));
  }
  c.fillStyle = ACCENT;
  c.beginPath();
  c.arc(cx, cy, r > 20 ? 4.5 : 3, 0, Math.PI * 2);
  c.fill();
}

function clearScene(c: CanvasRenderingContext2D) {
  c.clearRect(0, 0, W, H); // transparent: only the chair is drawn
}

function drawChair(c: CanvasRenderingContext2D, x: number, wheelAngle: number, tilt: number) {
  // soft ground shadow
  c.fillStyle = "rgba(27,37,64,0.18)";
  c.beginPath();
  c.ellipse(x + 26, GROUND + 5, 66, 8, 0, 0, Math.PI * 2);
  c.fill();

  c.save();
  c.translate(x, GROUND);
  c.rotate(tilt);
  c.lineCap = "round";
  c.lineJoin = "round";

  // person
  c.strokeStyle = BODY;
  c.lineWidth = 15; // torso
  line(c, -4, -60, -14, -106);
  c.lineWidth = 11; // thigh
  line(c, -2, -60, 38, -62);
  c.lineWidth = 9; // shin
  line(c, 38, -62, 54, -26);
  c.strokeStyle = INK;
  c.lineWidth = 6; // foot
  line(c, 54, -24, 68, -24);
  c.fillStyle = SKIN;
  c.beginPath();
  c.arc(-15, -124, 11, 0, Math.PI * 2); // head
  c.fill();
  c.strokeStyle = "#6f7ba0";
  c.lineWidth = 6; // arm
  c.beginPath();
  c.moveTo(-13, -102);
  c.lineTo(8, -84);
  c.lineTo(26, -82);
  c.stroke();

  // frame
  c.strokeStyle = ACCENT;
  c.lineWidth = 5;
  line(c, -20, -52, 42, -52); // seat
  line(c, -20, -52, -28, -112); // back rest
  line(c, -28, -112, -44, -114); // push handle
  line(c, -12, -80, 32, -80); // armrest
  line(c, 32, -80, 32, -52); // armrest post
  line(c, 42, -52, 58, -22); // footrest hanger
  line(c, 58, -22, 74, -22); // footplate
  line(c, -20, -52, 0, -34);
  line(c, 0, -34, 62, -34); // lower frame
  line(c, 42, -52, 62, -34);
  line(c, 62, -34, 62, -20); // caster fork

  // wheels
  wheel(c, 0, -REAR_R, REAR_R, wheelAngle, 10);
  wheel(c, 62, -11, 11, wheelAngle * 3, 4);

  c.restore();
}

export default function Wheelchair() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sim = useRef({
    x: 90,
    wheel: 0,
    tilt: 0,
    command: "stop" as Command,
    confidence: 0,
    lastMsg: 0,
    source: "idle",
  });
  const [status, setStatus] = useState("Not connected. Use the arrow keys, or connect live EEG.");

  function setCommand(command: Command, confidence: number, source: string) {
    sim.current.command = command;
    sim.current.confidence = confidence;
    sim.current.lastMsg = performance.now();
    sim.current.source = source;
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function draw(c: CanvasRenderingContext2D) {
      const s = sim.current;
      clearScene(c);
      drawChair(c, s.x, s.wheel, s.tilt);

      c.fillStyle = INK;
      c.font = "14px monospace";
      c.fillText(`${s.command.toUpperCase()}  ${(s.confidence * 100).toFixed(0)}%  (${s.source})`, 20, 26);
      c.strokeStyle = INK;
      c.lineWidth = 1;
      c.strokeRect(W - 150, 16, 120, 10); // confidence bar
      c.fillStyle = ACCENT;
      c.fillRect(W - 150, 16, 120 * s.confidence, 10);
    }

    let last = performance.now();
    let raf = 0;
    function frame(now: number) {
      const dt = (now - last) / 1000;
      last = now;
      const s = sim.current;
      if (now - s.lastMsg > FAILSAFE_MS) s.command = "stop"; // failsafe
      const dir = s.command === "forward" ? 1 : s.command === "back" ? -1 : 0;
      const nextX = Math.max(40, Math.min(W - 40, s.x + dir * SPEED * s.confidence * dt));
      s.wheel += (nextX - s.x) / REAR_R; // wheels turn with distance travelled
      s.x = nextX;
      const target = -dir * s.confidence * 0.035; // lean back slightly when moving forward
      s.tilt += (target - s.tilt) * Math.min(1, dt * 6);
      draw(ctx as CanvasRenderingContext2D);
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "ArrowRight") { e.preventDefault(); setCommand("forward", 1, "keyboard"); }
      if (e.key === "ArrowLeft") { e.preventDefault(); setCommand("back", 1, "keyboard"); }
      if (e.key === " " || e.key === "Escape") { e.preventDefault(); setCommand("stop", 0, "keyboard"); }
    }
    function onKeyUp(e: KeyboardEvent) {
      if (sim.current.source === "keyboard" && e.key.startsWith("Arrow")) setCommand("stop", 0, "keyboard");
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      wsRef.current?.close();
    };
  }, []);

  function connect() {
    wsRef.current?.close();
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => setStatus("Connected to EEG bridge.");
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      setCommand(m.command as Command, m.confidence, "EEG");
    };
    const lost = () => {
      wsRef.current = null;
      setCommand("stop", 0, "idle");
      setStatus("Not connected. Start eeg_bridge.py, then click Connect again.");
    };
    ws.onclose = lost;
    ws.onerror = lost;
  }

  return (
    <section style={{ maxWidth: 900, margin: "0 auto", padding: "32px 20px" }}>
      <h2>Wheelchair simulation</h2>
      <p>
        The chair moves from the commands the classifier sends: forward, back, or stop. If the signal is
        weak, it stops instead of guessing.
      </p>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        style={{ width: "100%", border: "1px solid #cfd6e2", borderRadius: 8 }}
        aria-label="Wheelchair simulation track"
      />
      <p aria-live="polite">{status}</p>
      <button onClick={connect}>Connect live EEG</button>
      <p style={{ fontSize: "0.9rem", opacity: 0.8 }}>
        Keyboard fallback: Right arrow = forward, Left arrow = back, Space = stop. Simulation only, not a
        medical device.
      </p>
    </section>
  );
}