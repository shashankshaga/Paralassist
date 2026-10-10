import { useEffect, useRef } from "react";

type Props = {
  samples: number[];
  color?: string;
};

export function EegTrace({ samples, color = "#3ee0b2" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c1210";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(62, 224, 178, 0.12)";
    ctx.lineWidth = 1;
    for (let y = 0; y < height; y += 16) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    if (samples.length < 2) return;
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    const span = Math.max(max - min, 0.4);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    samples.forEach((v, i) => {
      const x = (i / (samples.length - 1)) * width;
      const y = height - ((v - min) / span) * (height - 12) - 6;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [samples, color]);

  return <canvas ref={canvasRef} width={640} height={140} className="eeg-canvas" />;
}
