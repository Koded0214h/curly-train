import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchModelInfo, streamGenerate } from "./api";
import Visualizer from "./Visualizer";

const DEFAULT_PROMPT = "Ayanokoji-kun, what do you think the class should do next?";

function ModelBadge({ label, value, accent }) {
  return (
    <div className="badge">
      <span className="badge-label">{label}</span>
      <span className={`badge-value${accent ? " badge-accent" : ""}`}>{value}</span>
    </div>
  );
}

function Slider({ label, value, min, max, step, format, onChange }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="slider-label">
      <div className="slider-header">
        <span>{label}</span>
        <span className="slider-val">{format ? format(value) : value}</span>
      </div>
      <div className="slider-track">
        <div className="slider-fill" style={{ width: `${pct}%` }} />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
    </label>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [maxNewTokens, setMaxNewTokens] = useState(120);

  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [streamText, setStreamText] = useState("");
  const [tokenCount, setTokenCount] = useState(0);
  const [done, setDone] = useState(false);
  const [runMeta, setRunMeta] = useState(null);

  const abortRef = useRef(null);
  const outputRef = useRef(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setModelInfo)
      .catch((e) => setError(e.message));
  }, []);

  // auto-scroll output as text streams in
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [streamText]);

  const vizText = useMemo(
    () => (streamText ? streamText : prompt),
    [streamText, prompt],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setLoading(false);
  }, []);

  const onSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (loading) { stop(); return; }

    setLoading(true);
    setError("");
    setStreamText("");
    setTokenCount(0);
    setDone(false);
    setRunMeta(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamGenerate(
        { prompt, temperature, top_k: topK, max_new_tokens: maxNewTokens },
        (event) => {
          if (event.type === "token") {
            setStreamText(event.full_text);
            setTokenCount(event.generated_token_count);
          } else if (event.type === "done") {
            setStreamText(event.full_text);
            setTokenCount(event.generated_token_count);
            setRunMeta(event);
            setDone(true);
            setLoading(false);
          } else if (event.type === "error") {
            setError(event.detail || "Stream error");
            setLoading(false);
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message || "Generation failed");
      setLoading(false);
    }
  }, [loading, prompt, temperature, topK, maxNewTokens, stop]);

  const copyOutput = useCallback(() => {
    if (streamText) navigator.clipboard.writeText(streamText);
  }, [streamText]);

  const clearOutput = useCallback(() => {
    stop();
    setStreamText("");
    setTokenCount(0);
    setDone(false);
    setRunMeta(null);
    setError("");
  }, [stop]);

  const continuationText = useMemo(() => {
    if (!streamText) return "";
    const trimmed = prompt.trim();
    const idx = streamText.indexOf(trimmed);
    return idx === -1 ? streamText : streamText.slice(idx + trimmed.length).trimStart();
  }, [streamText, prompt]);

  return (
    <div className="shell">
      <div className="glow glow-a" />
      <div className="glow glow-b" />
      <div className="glow glow-c" />

      {/* ── header ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="brand-dot" />
          <span className="brand-name">Curly Train</span>
        </div>
        <div className="topbar-badges">
          <ModelBadge label="Checkpoint" value={modelInfo?.checkpoint ?? "…"} />
          <ModelBadge label="Device" value={modelInfo?.device ?? "…"} accent />
          <ModelBadge label="Context" value={modelInfo?.config?.block_size ? `${modelInfo.config.block_size} tok` : "…"} />
          <ModelBadge label="Vocab" value={modelInfo?.config?.vocab_size ? `${modelInfo.config.vocab_size.toLocaleString()}` : "…"} />
        </div>
      </header>

      {/* ── body ── */}
      <div className="body">

        {/* left: controls */}
        <aside className="panel controls-panel">
          <div className="panel-head">
            <span className="panel-title">Prompt</span>
            {streamText && (
              <button className="icon-btn" onClick={clearOutput} title="Clear output">✕</button>
            )}
          </div>

          <form onSubmit={onSubmit} className="controls-form">
            <textarea
              className="prompt-box"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={7}
              spellCheck="false"
              placeholder="Enter a prompt…"
            />

            <div className="sliders">
              <Slider label="Temperature" value={temperature} min={0.2} max={1.8} step={0.05}
                format={(v) => v.toFixed(2)} onChange={setTemperature} />
              <Slider label="Top-k" value={topK} min={5} max={200} step={5}
                onChange={setTopK} />
              <Slider label="Max tokens" value={maxNewTokens} min={32} max={512} step={8}
                onChange={setMaxNewTokens} />
            </div>

            <div className="btn-row">
              <button type="submit" className={loading ? "btn btn-stop" : "btn btn-primary"}>
                {loading ? (
                  <><span className="spinner" /> Stop</>
                ) : "Generate"}
              </button>
              {streamText && (
                <button type="button" className="btn btn-ghost" onClick={copyOutput}>
                  Copy
                </button>
              )}
            </div>

            {error && <div className="error-pill">{error}</div>}
          </form>
        </aside>

        {/* centre: visualizer */}
        <section className="viz-card">
          <div className="viz-top-bar">
            <span className="viz-label">Token Graph</span>
            {loading && (
              <span className="live-pill">
                <span className="pulse-dot" /> Live · {tokenCount} tok
              </span>
            )}
            {done && !loading && (
              <span className="done-pill">{tokenCount} tokens</span>
            )}
          </div>
          <Visualizer text={vizText} />
        </section>

        {/* right: output */}
        <aside className="panel output-panel">
          <div className="panel-head">
            <span className="panel-title">Output</span>
            {done && <span className="tag-done">Done</span>}
          </div>

          <div ref={outputRef} className="output-scroll">
            {streamText ? (
              <div className="output-text">
                <span className="output-prompt">{prompt.trim()}</span>
                {continuationText && (
                  <span className="output-continuation">{" "}{continuationText}</span>
                )}
                {loading && <span className="cursor" />}
              </div>
            ) : (
              <div className="output-empty">
                Generation will stream here token by token.
              </div>
            )}
          </div>

          {runMeta && (
            <div className="meta-grid">
              <div className="meta-row">
                <span>Prompt tokens</span>
                <strong>{runMeta.config ? tokenCount - maxNewTokens : "—"}</strong>
              </div>
              <div className="meta-row">
                <span>Total tokens</span>
                <strong>{tokenCount}</strong>
              </div>
              <div className="meta-row">
                <span>Temperature</span>
                <strong>{temperature.toFixed(2)}</strong>
              </div>
              <div className="meta-row">
                <span>Top-k</span>
                <strong>{topK}</strong>
              </div>
              <div className="meta-row">
                <span>Checkpoint</span>
                <strong className="meta-mono">{runMeta.checkpoint}</strong>
              </div>
            </div>
          )}

          {!runMeta && !loading && (
            <div className="meta-hint">Run details appear here after generation.</div>
          )}
        </aside>

      </div>
    </div>
  );
}
