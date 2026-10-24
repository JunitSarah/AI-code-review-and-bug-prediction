import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { analyzePR, analyzeSnippet } from "./api";

const riskColor = (r) => (r >= 0.7 ? "#d64545" : r >= 0.5 ? "#e08a1e" : "#2f9e63");

function RiskBar({ value }) {
  return (
    <div className="bar" title={`${(value * 100).toFixed(0)}% defect-prone`}>
      <div className="fill" style={{ width: `${value * 100}%`, background: riskColor(value) }} />
      <span>{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function FileCard({ file }) {
  const [open, setOpen] = useState(file.max_risk >= 0.5);
  return (
    <div className="card">
      <button className="file-head" onClick={() => setOpen(!open)}>
        <span className="fname">{file.filename}</span>
        <RiskBar value={file.max_risk} />
      </button>
      {open &&
        file.hunks.map((h) => (
          <div key={h.hunk_index} className="hunk">
            <div className="hunk-meta">
              Hunk {h.hunk_index + 1} · <b style={{ color: riskColor(h.risk) }}>{h.label}</b>
              {h.rank_percentile >= 0.8 && <span className="top"> · top {Math.max(1, Math.round((1 - h.rank_percentile) * 100))}% riskiest in this PR</span>}
              <RiskBar value={h.risk} />
            </div>
            <pre>{h.snippet}</pre>
          </div>
        ))}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("pr");
  const [prUrl, setPrUrl] = useState("");
  const [code, setCode] = useState("");
  const [language, setLanguage] = useState("python");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function run(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(mode === "pr" ? await analyzePR(prUrl) : await analyzeSnippet(code, language));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>AI Code Review &amp; Bug Prediction</h1>
      <p className="sub">CodeBERT fine-tuned on Defects4J + BugsInPy, with an LLM-written review.</p>
      <p className="note">
        Scores rank changes by <b>relative</b> defect risk to help you decide what to review first.
        They are not proof that a bug exists, and a low score is not proof that code is correct.
      </p>

      <div className="tabs">
        <button className={mode === "pr" ? "on" : ""} onClick={() => setMode("pr")}>GitHub PR</button>
        <button className={mode === "snippet" ? "on" : ""} onClick={() => setMode("snippet")}>Code snippet</button>
      </div>

      <form onSubmit={run}>
        {mode === "pr" ? (
          <input value={prUrl} onChange={(e) => setPrUrl(e.target.value)} required
                 placeholder="https://github.com/owner/repo/pull/123" />
        ) : (
          <>
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option>python</option>
              <option>java</option>
            </select>
            <textarea rows={12} value={code} onChange={(e) => setCode(e.target.value)} required
                      placeholder="Paste code here…" />
          </>
        )}
        <button className="primary" disabled={loading}>{loading ? "Analyzing…" : "Analyze"}</button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <section>
          <div className="overall">
            Overall risk <RiskBar value={result.overall_risk} />
          </div>
          {result.summary_error && <div className="note">{result.summary_error}</div>}
          {result.summary && (
            <div className="card summary">
              <h2>Review</h2>
              <ReactMarkdown>{result.summary}</ReactMarkdown>
            </div>
          )}
          <h2>Files by risk</h2>
          {result.files.map((f) => <FileCard key={f.filename} file={f} />)}
        </section>
      )}
    </main>
  );
}
