import { useEffect, useRef, useState } from "react";

// Wheelchair simulation. Receives {"command": "forward"|"back"|"stop", "confidence": 0-1}
// from eeg_bridge.py over WebSocket. Arrow keys work as a fallback (Space/Esc = stop).

const WS_URL = "ws://localhost:8765";
const SPEED = 120; // pixels per second at full confidence
const FAILSAFE_MS = 1500; // stop if no message arrives for this long
const W = 800;
const H = 220;

type Command = "forward" | "back" | "stop";

export default function Wheelchair() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sim = useRef({ x: 60, command: "stop" as Command, confidence: 0, lastMsg: 0, source: "idle" });
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

    const ink = "#1b2540";
    const accent = "#3b4cca";

    function draw() {
      const s = sim.current;
      ctx!.fillStyle = "#f4f6fa";
      ctx!.fillRect(0, 0, W, H);
      ctx!.strokeStyle = ink;
      ctx!.lineWidth = 2;
      ctx!.beginPath();
      ctx!.moveTo(20, H - 40);
      ctx!.lineTo(W - 20, H - 40);
      ctx!.stroke();
      ctx!.fillStyle = ink;
      ctx!.font = "14px monospace";
      ctx!.fillText("START", 20, H - 18);
      ctx!.fillText("END", W - 55, H - 18);

      const x = s.x;
      const y = H - 40;
      ctx!.strokeStyle = accent;
      ctx!.fillStyle = accent;
      ctx!.lineWidth = 4;
      ctx!.beginPath();
      ctx!.arc(x, y - 22, 22, 0, Math.PI * 2); // wheel
      ctx!.stroke();
      ctx!.beginPath();
      ctx!.moveTo(x - 10, y - 30);
      ctx!.lineTo(x - 10, y - 70);
      ctx!.lineTo(x - 30, y - 70); // back rest
      ctx!.stroke();
      ctx!.beginPath();
      ctx!.moveTo(x - 10, y - 46);
      ctx!.lineTo(x + 28, y - 46); // seat
      ctx!.stroke();
      ctx!.beginPath();
      ctx!.arc(x - 20, y - 86, 8, 0, Math.PI * 2); // head
      ctx!.fill();

      ctx!.fillStyle = ink;
      ctx!.fillText(`${s.command.toUpperCase()}  ${(s.confidence * 100).toFixed(0)}%  (${s.source})`, 20, 24);
    }

    let last = performance.now();
    let raf = 0;
    function frame(now: number) {
      const dt = (now - last) / 1000;
      last = now;
      const s = sim.current;
      if (now - s.lastMsg > FAILSAFE_MS) s.command = "stop"; // failsafe
      const dir = s.command === "forward" ? 1 : s.command === "back" ? -1 : 0;
      s.x = Math.max(40, Math.min(W - 40, s.x + dir * SPEED * s.confidence * dt));
      draw();
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
