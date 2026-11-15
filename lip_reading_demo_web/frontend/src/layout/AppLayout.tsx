import { useState } from "react";
import { Outlet } from "react-router-dom";
import "./AppLayout.css";

function Logo({ name, text }: { name: string; text: string }) {
  const [available, setAvailable] = useState(true);
  return available
    ? <img src={`/assets/${name}.png`} alt={text} onError={() => setAvailable(false)} />
    : <span className="sol10-logo-text">{text}</span>;
}

export default function App() {
  return (
    <div className="sol10-app">
      <div className="sol10-spectrum" aria-hidden="true" />
      <header className="sol10-header">
        <div className="sol10-logo-left"><Logo name="logo-aiml" text="AI MultimediaLab" /></div>
        <div className="sol10-logo-center"><Logo name="logo-sol10" text="Lip reading module" /></div>
        <div className="sol10-logo-right"><Logo name="logo-upb" text="UNSTPB" /></div>
      </header>
      <main className="sol10-content"><Outlet /></main>
    </div>
  );
}
