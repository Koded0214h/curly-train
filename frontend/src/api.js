// Empty string = relative URLs → Vite proxy forwards /api/* to backend in dev,
// and FastAPI serves them directly from the same origin in production.
export const API_BASE = import.meta.env.VITE_API_URL || "";

export async function fetchModelInfo() {
  const res = await fetch(`${API_BASE}/api/model`);
  if (!res.ok) throw new Error(`Model info failed: ${res.status}`);
  return res.json();
}

export async function generateText(payload) {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Generation failed: ${res.status}`);
  }
  return res.json();
}

// Streams SSE events from /api/generate/stream, calling onEvent for each
export async function streamGenerate(payload, onEvent, signal) {
  const params = new URLSearchParams({
    prompt: payload.prompt,
    temperature: payload.temperature,
    top_k: payload.top_k,
    max_new_tokens: payload.max_new_tokens,
  });
  const res = await fetch(`${API_BASE}/api/generate/stream?${params}`, { signal });
  if (!res.ok) throw new Error(`Stream failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      if (part.startsWith("data: ")) {
        try { onEvent(JSON.parse(part.slice(6))); } catch { /* skip malformed */ }
      }
    }
  }
}
