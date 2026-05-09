const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function fetchModelInfo() {
  const response = await fetch(`${API_BASE}/api/model`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.status}`);
  }
  return response.json();
}

export async function generateText(payload) {
  const response = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Generation failed: ${response.status}`);
  }

  return response.json();
}

