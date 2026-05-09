import { useEffect, useMemo, useState } from "react";
import { fetchModelInfo, generateText } from "./api";
import Visualizer from "./Visualizer";

const DEFAULT_PROMPT =
  "Ayanokoji-kun, what do you think the class should do next?";

export default function App() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [maxNewTokens, setMaxNewTokens] = useState(120);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelInfo, setModelInfo] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    let active = true;
    fetchModelInfo()
      .then((data) => {
        if (active) {
          setModelInfo(data);
        }
      })
      .catch((err) => {
        if (active) {
          setError(err.message);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const displayText = useMemo(() => {
    if (!result) {
      return prompt;
    }
    return `${prompt.trim()} ${result.continuation_text}`.trim();
  }, [prompt, result]);

  const onSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const data = await generateText({
        prompt,
        temperature,
        top_k: topK,
        max_new_tokens: maxNewTokens,
      });
      setResult(data);
    } catch (err) {
      setError(err.message || "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  const copyOutput = async () => {
    if (!result?.full_text) {
      return;
    }
    await navigator.clipboard.writeText(result.full_text);
  };

  return (
    <div className="app-shell">
      <div className="backdrop backdrop-a" />
      <div className="backdrop backdrop-b" />

      <main className="layout">
        <section className="hero">
          <div className="eyebrow">Curly Train</div>
          <h1>Dialogue generation with a live 3D token field.</h1>
          <p>
            Type a prompt, generate a continuation from the local transformer,
            and watch the output collapse into a moving subword sculpture.
          </p>
        </section>

        <section className="status-row">
          <div className="status-card">
            <span className="status-label">Checkpoint</span>
            <strong>{modelInfo?.checkpoint || "loading..."}</strong>
          </div>
          <div className="status-card">
            <span className="status-label">Device</span>
            <strong>{modelInfo?.device || "loading..."}</strong>
          </div>
          <div className="status-card">
            <span className="status-label">Context</span>
            <strong>{modelInfo?.config?.block_size || "..."}</strong>
          </div>
        </section>

        <section className="content-grid">
          <form className="control-panel" onSubmit={onSubmit}>
            <label>
              Prompt
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={8}
                spellCheck="false"
              />
            </label>

            <div className="slider-grid">
              <label>
                Temperature: {temperature.toFixed(2)}
                <input
                  type="range"
                  min="0.2"
                  max="1.4"
                  step="0.05"
                  value={temperature}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                />
              </label>

              <label>
                Top-k: {topK}
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="1"
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>

              <label>
                Max tokens: {maxNewTokens}
                <input
                  type="range"
                  min="32"
                  max="256"
                  step="8"
                  value={maxNewTokens}
                  onChange={(event) => setMaxNewTokens(Number(event.target.value))}
                />
              </label>
            </div>

            <div className="button-row">
              <button type="submit" disabled={loading}>
                {loading ? "Generating..." : "Generate"}
              </button>
              <button type="button" className="secondary" onClick={copyOutput}>
                Copy output
              </button>
            </div>

            {error ? <div className="error-box">{error}</div> : null}
          </form>

          <section className="visual-card">
            <Visualizer text={displayText} />
            <div className="visual-overlay">
              <div className="overlay-title">Token Space</div>
              <div className="overlay-text">
                {result ? result.full_text : "Waiting for a generation run."}
              </div>
            </div>
          </section>
        </section>

        <section className="output-grid">
          <article className="output-card">
            <h2>Generated Text</h2>
            <pre>{result?.full_text || "No output yet."}</pre>
          </article>

          <article className="output-card meta-card">
            <h2>Run Details</h2>
            <dl>
              <div>
                <dt>Prompt tokens</dt>
                <dd>{result?.prompt_token_count ?? "—"}</dd>
              </div>
              <div>
                <dt>Total tokens</dt>
                <dd>{result?.generated_token_count ?? "—"}</dd>
              </div>
              <div>
                <dt>Temperature</dt>
                <dd>{temperature.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Top-k</dt>
                <dd>{topK}</dd>
              </div>
            </dl>
          </article>
        </section>
      </main>
    </div>
  );
}

