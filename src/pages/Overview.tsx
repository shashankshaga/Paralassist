import { Brain } from "../components/Brain";

export function Overview() {
  return (
    <div className="page">
      <section className="hero hero-split">
        <div>
          <p className="kicker">Hackathon · BCI · Care</p>
          <h1>
            Paralassist
            <span>Look to speak.</span>
          </h1>
          <p className="lede">
            People with paralysis can still lock their visual cortex onto a flicker. Paralassist
            mirrors those SSVEP frequencies, classifies them with competing AI models, and turns
            a gaze into a need: help, water, pain, yes, no.
          </p>
        </div>
        <Brain />
      </section>

      <section className="grid-3">
        <article className="card">
          <h2>Problem</h2>
          <p>
            Locked-in syndrome, late-stage ALS, and high spinal injury can leave someone fully
            aware but unable to talk, type, or press a button. Instead of guessing, now we can be sure.
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
            We trained our calibrated data on a TRCA model, and at the same time trained a 
            model to classify the SSVEP frequencies using an EEGNet Neural Network and compare the two outputs.
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
