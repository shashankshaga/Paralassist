import { useMemo, useState } from "react";
import { NEEDS } from "../needs";
import { runBenchmark, type Benchmark } from "../signal";

function formatPct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

export function Lab() {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<Benchmark[] | null>(null);
  const [seed, setSeed] = useState(11);

  const best = useMemo(() => {
    if (!results?.length) return null;
    return [...results].sort((a, b) => b.accuracy - a.accuracy)[0];
  }, [results]);

  function train() {
    setRunning(true);
    window.setTimeout(() => {
      setResults(runBenchmark(seed));
      setRunning(false);
    }, 60);
  }

  return (
    <div className="page">
      <header className="board-head">
        <div>
          <p className="kicker">Model lab</p>
          <h1>Train on competing SSVEP models</h1>
          <p className="hint">
            Same synthetic corpus: 18 one-second trials per need, 256 Hz, harmonics at 2f and
            3f plus noise. CCA and FBCCA are unsupervised. FFT-ML is a softmax classifier
            trained in the browser on Goertzel band power.
          </p>
        </div>
        <div className="hero-actions">
          <label className="seed">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 1)}
            />
          </label>
          <button className="btn btn-primary" onClick={train} disabled={running}>
            {running ? "Training…" : "Run training"}
          </button>
        </div>
      </header>

      <section className="grid-3">
        <article className="card">
          <h2>CCA</h2>
          <p>
            Canonical correlation against sine/cosine at each stimulus frequency. Fast, little
            data, the classical SSVEP baseline.
          </p>
        </article>
        <article className="card">
          <h2>FBCCA</h2>
          <p>
            Filter-bank CCA: same idea, but scores f, 2f, and 3f together. Better when the
            brain rings at harmonics.
          </p>
        </article>
        <article className="card">
          <h2>FFT-ML</h2>
          <p>
            Learns weights over frequency features. Stands in for EEGNet / CNN training when
            we later swap in recorded SSVEP datasets.
          </p>
        </article>
      </section>

      {results && (
        <>
          <section className="card">
            <h2>Hold-out results</h2>
            {best && (
              <p className="hint">
                Best accuracy this run: <strong>{best.name}</strong> at {formatPct(best.accuracy)}.
              </p>
            )}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>ITR (bits/min)</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row) => (
                    <tr key={row.name} className={row.name === best?.name ? "best" : ""}>
                      <td>{row.name}</td>
                      <td>{formatPct(row.accuracy)}</td>
                      <td>{row.itr.toFixed(1)}</td>
                      <td>{row.latencyMs} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {results.map((row) => (
            <section className="card" key={`${row.name}-detail`}>
              <h2>
                {row.name} confusion
                {row.losses && (
                  <span className="loss">
                    final loss {row.losses[row.losses.length - 1]?.toFixed(3)}
                  </span>
                )}
              </h2>
              {row.losses && (
                <div className="loss-bars" title="Training loss by epoch">
                  {row.losses.filter((_, i) => i % 4 === 0).map((loss, i) => (
                    <span key={i} style={{ height: `${Math.max(8, (1.2 - loss) * 70)}px` }} />
                  ))}
                </div>
              )}
              <div className="table-wrap">
                <table className="confusion">
                  <thead>
                    <tr>
                      <th />
                      {NEEDS.map((need) => (
                        <th key={need.id}>{need.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {NEEDS.map((need, i) => (
                      <tr key={need.id}>
                        <th>{need.label}</th>
                        {row.confusion[i].map((cell, j) => (
                          <td
                            key={j}
                            style={{
                              background: `rgba(62, 224, 178, ${
                                cell / Math.max(1, Math.max(...row.confusion[i]))
                              })`,
                            }}
                          >
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ))}
        </>
      )}
    </div>
  );
}
