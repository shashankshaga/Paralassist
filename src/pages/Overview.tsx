import { Brain } from "../components/Brain";

type Props = {
  onOpenBoard: () => void;
  onOpenLab: () => void;
};

export function Overview({ onOpenBoard, onOpenLab }: Props) {
  return (
    <div className="page">
      <section className="hero hero-split">
        <div>
          <p className="kicker">Hackathon · BCI · Care</p>
          <h1>
            StillSpeak
            <span>Look to speak.</span>
          </h1>
          <p className="lede">
            People with paralysis can still lock their visual cortex onto a flicker. StillSpeak
            mirrors those SSVEP frequencies, classifies them with competing AI models, and turns
            a gaze into a need: help, water, pain, yes, no.
          </p>
          <div className="hero-actions">
            <button className="btn btn-primary" onClick={onOpenBoard}>
              Open patient board
            </button>
            <button className="btn" onClick={onOpenLab}>
              Train SSVEP models
            </button>
          </div>
        </div>
        <Brain />
      </section>

      <section className="grid-3">
        <article className="card">
          <h2>Problem</h2>
          <p>
            Locked-in syndrome, late-stage ALS, and high spinal injury can leave someone fully
            aware but unable to talk, type, or press a button. Care teams guess. Patients wait.
          </p>
        </article>
        <article className="card">
          <h2>SSVEP lock</h2>
          <p>
            A tile flashing at 12 Hz drives a 12 Hz rhythm (plus harmonics) in the occipital
            lobe — visual cortex. That frequency is a selectable channel. No movement required —
            only gaze.
          </p>
        </article>
        <article className="card">
          <h2>AI layer</h2>
          <p>
            We train and compare CCA, filter-bank CCA, and an in-browser FFT + softmax model
            on synthetic SSVEP. The live board uses FBCCA on a 1-second window.
          </p>
        </article>
      </section>

      <section className="pipeline">
        <h2>Pipeline</h2>
        <ol>
          <li>
            <strong>Stimulate</strong>
            <span>Eight needs, eight frequencies, 7.5–15 Hz square-wave flicker.</span>
          </li>
          <li>
            <strong>Acquire</strong>
            <span>EEG at 256 Hz. Demo uses a synthetic visual-cortex mixture + noise.</span>
          </li>
          <li>
            <strong>Classify</strong>
            <span>Score each target frequency. Dwell until confidence holds, then speak the phrase.</span>
          </li>
        </ol>
      </section>
    </div>
  );
}
