"""
eeg_bridge.py: reads live EEG, classifies which flashing target the person is
looking at, and broadcasts a wheelchair command to the website over WebSocket.

Install:  pip install pylsl websockets numpy scipy scikit-learn
Run live: python eeg_bridge.py
Record:   python eeg_bridge.py --record session.jsonl
Replay:   python eeg_bridge.py --replay session.jsonl   (no hardware needed)
"""
import argparse, asyncio, json, time
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
import websockets

# ---------- CONFIG ----------
FREQS = {"forward": 10.0, "back": 12.0}   # must match the flicker on your stimulus screen
WINDOW_SEC = 3.0
SEND_EVERY = 0.5
THRESHOLD = 0.6        # below this -> "stop" (do not guess)
CONFIRMATIONS = 2      # same answer this many times in a row before moving
PORT = 8765


def bandpass(x, fs, lo=5, hi=40):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def reference(freq, fs, n, harmonics=2):
    t = np.arange(n) / fs
    return np.column_stack([f(2 * np.pi * h * freq * t)
                            for h in range(1, harmonics + 1) for f in (np.sin, np.cos)])


def classify_cca(window, fs):
    """Baseline: returns {command: correlation}."""
    scores = {}
    for name, f in FREQS.items():
        ref = reference(f, fs, window.shape[0])
        cca = CCA(n_components=1).fit(window, ref)
        u, v = cca.transform(window, ref)
        scores[name] = abs(np.corrcoef(u[:, 0], v[:, 0])[0, 1])
    return scores


def classify_model(window, fs):
    """PLUG IN YOUR TRAINED MODEL HERE.
    Return {command: probability}. Example with a saved PyTorch model:
        x = torch.tensor(window.T[None], dtype=torch.float32)
        p = torch.softmax(model(x), dim=1)[0].numpy()
        return {"forward": float(p[0]), "back": float(p[1])}
    """
    return classify_cca(window, fs)  # falls back to CCA until you connect your model


def decide(scores):
    best = max(scores, key=scores.get)
    conf = scores[best]
    return (best if conf >= THRESHOLD else "stop"), conf


async def live_source(record_path):
    from pylsl import StreamInlet, resolve_byprop
    streams = resolve_byprop("type", "EEG", timeout=10)
    if not streams:
        raise SystemExit("No EEG stream found. Check that your device is streaming via LSL.")
    inlet = StreamInlet(streams[0])
    fs = inlet.info().nominal_srate()
    n = int(WINDOW_SEC * fs)
    buf = np.zeros((0, inlet.info().channel_count()))
    rec = open(record_path, "a") if record_path else None
    while True:
        chunk, _ = inlet.pull_chunk(timeout=0.1)
        if chunk:
            buf = np.vstack([buf, np.array(chunk)])[-n:]
        if buf.shape[0] >= n:
            scores = classify_model(bandpass(buf, fs), fs)
            cmd, conf = decide(scores)
            msg = {"command": cmd, "confidence": round(conf, 3), "t": time.time()}
            if rec:
                rec.write(json.dumps(msg) + "\n"); rec.flush()
            yield msg
        await asyncio.sleep(SEND_EVERY)


async def replay_source(path):
    msgs = [json.loads(l) for l in open(path)]
    while True:
        for m in msgs:
            yield m
            await asyncio.sleep(SEND_EVERY)


async def main(args):
    clients = set()

    async def handler(ws):
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)

    source = replay_source(args.replay) if args.replay else live_source(args.record)
    async with websockets.serve(handler, "localhost", PORT):
        print(f"Bridge running on ws://localhost:{PORT}")
        last, streak = "stop", 0
        async for msg in source:
            # require repeated agreement before sending a movement command
            streak = streak + 1 if msg["command"] == last else 1
            last = msg["command"]
            if msg["command"] != "stop" and streak < CONFIRMATIONS:
                msg = {**msg, "command": "stop"}
            websockets.broadcast(clients, json.dumps(msg))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record"); p.add_argument("--replay")
    asyncio.run(main(p.parse_args()))
