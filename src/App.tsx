import { useState } from "react";
import { Overview } from "./pages/Overview";
import Wheelchair from "./components/Wheelchair";

type Tab = "overview" | "wheelchair";

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
            Story
          </button>
          <button className={tab === "wheelchair" ? "on" : ""} onClick={() => setTab("wheelchair")}>
            Wheelchair
          </button>
        </div>
      </nav>
      {tab === "overview" && (
        <Overview
          onOpenBoard={() => setTab("wheelchair")}
          onOpenLab={() => setTab("wheelchair")}
        />
      )}
      {tab === "wheelchair" && <Wheelchair />}
      <footer className="foot">
        Research demo · synthetic EEG · not a medical device
      </footer>
    </div>
  );
}
