import { useState } from "react";
import { Overview } from "./pages/Overview";
import { Board } from "./pages/Board";
import { Lab } from "./pages/Lab";
import Wheelchair from "./components/Wheelchair";

type Tab = "overview" | "board" | "lab" | "wheelchair";

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");

  return (
    <div className="shell">
      <nav className="nav">
        <div className="brand">
          <span className="mark" />
          StillSpeak
        </div>
        <div className="tabs">
          <button className={tab === "overview" ? "on" : ""} onClick={() => setTab("overview")}>
            Story
          </button>
          <button className={tab === "board" ? "on" : ""} onClick={() => setTab("board")}>
            Patient board
          </button>
          <button className={tab === "lab" ? "on" : ""} onClick={() => setTab("lab")}>
            Train models
          </button>
          <button className={tab === "wheelchair" ? "on" : ""} onClick={() => setTab("wheelchair")}>
            Wheelchair
          </button>
        </div>
      </nav>
      {tab === "overview" && (
        <Overview onOpenBoard={() => setTab("board")} onOpenLab={() => setTab("lab")} />
      )}
      {tab === "board" && <Board />}
      {tab === "lab" && <Lab />}
      {tab === "wheelchair" && <Wheelchair />}
      <footer className="foot">
        Research demo · synthetic EEG · not a medical device
      </footer>
    </div>
  );
}
