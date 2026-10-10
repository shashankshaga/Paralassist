import { useEffect, useRef, type CSSProperties } from "react";
import type { Need } from "../needs";

type Props = {
  need: Need;
  stimOn: boolean;
  attending: boolean;
  confidence: number;
  onAttend: (id: Need["id"] | null) => void;
};

export function FlickerTile({ need, stimOn, attending, confidence, onAttend }: Props) {
  const panelRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    if (!stimOn) {
      el.style.setProperty("--flash", "0");
      return;
    }
    let frame = 0;
    const start = performance.now();
    const loop = (now: number) => {
      const t = (now - start) / 1000;
      const on = (t * need.hz) % 1 < 0.5;
      el.style.setProperty("--flash", on ? "1" : "0");
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [stimOn, need.hz]);

  return (
    <button
      ref={panelRef}
      className={`tile ${attending ? "tile-hot" : ""}`}
      style={{ "--accent": need.accent } as CSSProperties}
      onPointerEnter={() => onAttend(need.id)}
      onPointerLeave={() => onAttend(null)}
      onFocus={() => onAttend(need.id)}
      onBlur={() => onAttend(null)}
    >
      <span className="tile-flash" />
      <span className="tile-body">
        <span className="tile-hz">{need.hz.toFixed(2)} Hz</span>
        <span className="tile-label">{need.label}</span>
        <span className="meter">
          <span style={{ width: `${Math.round(confidence * 100)}%` }} />
        </span>
      </span>
    </button>
  );
}
