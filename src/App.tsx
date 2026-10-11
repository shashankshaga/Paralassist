import { useState } from "react";
import { Overview } from "./pages/Overview";
import { Detector } from "./pages/Detector";
import Wheelchair from "./components/Wheelchair";

type Tab = "overview" | "wheelchair" | "detector";

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");

  return (
    <div className="shell">
      <nav className="nav">
        <div className="brand">
          <span className="mark" />
          Paralassist
        </div>
        <div className="tabs">
          <button className={tab === "overview" ? "on" : ""} onClick={() => setTab("overview")}>
            Home
          </button>
          <button className={tab === "wheelchair" ? "on" : ""} onClick={() => setTab("wheelchair")}>
            Wheelchair
          </button>
          <button className={tab === "detector" ? "on" : ""} onClick={() => setTab("detector")}>
            Detector
          </button>
        </div>
      </nav>
      {tab === "overview" && <Overview />}
      {tab === "wheelchair" && <Wheelchair />}
      {tab === "detector" && <Detector />}
      <footer className="foot">
        Research demo · synthetic EEG · not a medical device
      </footer>
    </div>
  );
}