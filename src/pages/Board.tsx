import { useEffect, useMemo, useRef, useState } from "react";
import { FlickerTile } from "../components/FlickerTile";
import { EegTrace } from "../components/EegTrace";
import { NEEDS, FS, WINDOW_SAMPLES, type NeedId } from "../needs";
import { classifyFbcca, generateTrial } from "../signal";

type LogItem = { at: string; phrase: string; hz: number };

export function Board() {
  const [allowed, setAllowed] = useState(false);
  const [stimOn, setStimOn] = useState(false);
  const [attendId, setAttendId] = useState<NeedId | null>(null);
  const [confidences, setConfidences] = useState<number[]>(() => NEEDS.map(() => 0));
  const [trace, setTrace] = useState<number[]>(() => Array(WINDOW_SAMPLES).fill(0));
  const [spoken, setSpoken] = useState<string>("Waiting for a stable lock…");
  const [log, setLog] = useState<LogItem[]>([]);
  const dwell = useRef(0);
  const lastSpeak = useRef("");
  const seed = useRef(1);

  const attendHz = useMemo(
    () => NEEDS.find((n) => n.id === attendId)?.hz ?? null,
    [attendId],
  );

  useEffect(() => {
    if (!allowed || !stimOn) return;
    const id = window.setInterval(() => {
      const hz = attendHz ?? 0;
      const noise = attendHz ? 0.78 : 1.35;
      const x = generateTrial(hz || 6, noise, seed.current++);
      const result = classifyFbcca(x);
      setTrace(Array.from(x));
      setConfidences(result.scores.map((s) => Math.min(1, s * 1.15)));

      const locked = result.confidence > 0.34 && attendHz !== null;
      dwell.current = locked ? dwell.current + 1 : 0;
      if (dwell.current >= 3) {
        const need = NEEDS[result.index];
        if (need.phrase !== lastSpeak.current) {
          lastSpeak.current = need.phrase;
          setSpoken(need.phrase);
          setLog((prev) => [
            { at: new Date().toLocaleTimeString(), phrase: need.phrase, hz: need.hz },
            ...prev,
          ].slice(0, 8));
          window.speechSynthesis?.cancel();
          window.speechSynthesis?.speak(new SpeechSynthesisUtterance(need.phrase));
        }
        dwell.current = 0;
      }
    }, 280);
    return () => window.clearInterval(id);
  }, [allowed, stimOn, attendHz]);

  if (!allowed) {
    return (
      <div className="page">
        <section className="warn card">
          <p className="kicker">Photosensitivity</p>
          <h1>This demo flashes lights on purpose.</h1>
          <p>
            SSVEP needs rhythmic flicker. Do not enable stimulation if you have epilepsy,
            migraine with aura, or feel unwell with flashing lights. The board still works
            as a static layout if you skip flicker.
          </p>
          <div className="hero-actions">
            <button
              className="btn btn-primary"
              onClick={() => {
                setAllowed(true);
                setStimOn(true);
              }}
            >
              I understand — enable flicker
            </button>
            <button className="btn" onClick={() => setAllowed(true)}>
              Continue without flicker
            </button>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="page board-page">
      <header className="board-head">
        <div>
          <p className="kicker">Patient channel</p>
          <h1>Gaze a need. Hold it.</h1>
          <p className="hint">
            Hover a tile to simulate looking. Synthetic EEG is generated at that tile’s
            frequency, then FBCCA votes every ~280 ms. Three stable votes speak the phrase.
          </p>
        </div>
        <label className="switch">
          <input
            type="checkbox"
            checked={stimOn}
            onChange={(e) => setStimOn(e.target.checked)}
          />
          Visual stimulation {stimOn ? "on" : "off"}
        </label>
      </header>

      <div className="board-grid">
        {NEEDS.map((need, i) => (
          <FlickerTile
            key={need.id}
            need={need}
            stimOn={stimOn}
            attending={attendId === need.id}
            confidence={confidences[i] ?? 0}
            onAttend={setAttendId}
          />
        ))}
      </div>

      <section className="readout">
        <div>
          <p className="kicker">Decoded intent</p>
          <p className="spoken">{spoken}</p>
          <p className="mono">
            fs {FS} Hz · window {WINDOW_SAMPLES} samples · model FBCCA ·
            {attendHz ? ` attending ${attendHz} Hz` : " no gaze"}
          </p>
        </div>
        <EegTrace samples={trace} />
      </section>

      <section className="log card">
        <h2>Session log</h2>
        {log.length === 0 ? (
          <p>No lock yet. Hover Help or Water to drive a frequency into the signal.</p>
        ) : (
          <ul>
            {log.map((item, i) => (
              <li key={`${item.at}-${i}`}>
                <span>{item.at}</span>
                <strong>{item.phrase}</strong>
                <em>{item.hz} Hz</em>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
