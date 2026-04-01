const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin || 'http://localhost:8000';

async function handleResponse(res) {
  if (!res.ok) {
    let detail = 'Request failed';
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (_) {
      detail = res.statusText || detail;
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchModalities() {
  const res = await fetch(`${API_BASE}/modalities`);
  return handleResponse(res);
}

export async function runQuery(payload) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export { API_BASE };
