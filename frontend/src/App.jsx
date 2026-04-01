import { useEffect, useMemo, useState } from 'react';
import { fetchModalities, runQuery } from './api';

const samples = [
  { title: 'Land cover', text: 'What is the land cover in Punjab district?' },
  { title: 'Crop health', text: 'Analyze crop health and NDVI in Haryana for Rabi season 2022.' },
  { title: 'Flood status', text: 'Show flood extent near Guwahati for 2023.' },
  { title: 'Coordinates', text: 'Give land cover at latitude 28.6 and longitude 77.2.' },
];

const defaultModalities = ['climate', 's2', 's1', 'planet'];

const defaultQuery = 'What is the land cover in Punjab?';

function App() {
  const [query, setQuery] = useState(defaultQuery);
  const [availableMods, setAvailableMods] = useState(defaultModalities);
  const [selectedMods, setSelectedMods] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [mapDoc, setMapDoc] = useState('');

  useEffect(() => {
    fetchModalities()
      .then((res) => {
        const apiMods = res && res.available && res.available.length ? res.available : [];
        if (apiMods.length) {
          setAvailableMods(apiMods);
        }
      })
      .catch(() => {
        // keep defaultModalities already set
      });
  }, []);

  const modalityBadges = useMemo(() => {
    if (!result?.modality_counts) return [];
    return Object.entries(result.modality_counts)
      .filter(([, count]) => count > 0)
      .map(([name, count]) => ({ name, count }));
  }, [result]);

  const toggleModality = (mod) => {
    setSelectedMods((prev) => (prev.includes(mod) ? prev.filter((m) => m !== mod) : [...prev, mod]));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      setError('Enter a question first.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const payload = {
        query,
        modalities: selectedMods.length ? selectedMods : null,
        max_rows: 50000,
        heatmap: true,
      };
      const data = await runQuery(payload);
      setResult(data);
      setMapDoc(data.map_html || '');
    } catch (err) {
      setError(err.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const hasResult = Boolean(result);

  const applySample = (text) => {
    setQuery(text);
    const textarea = document.getElementById('query');
    if (textarea) {
      textarea.focus();
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-inline">
          <span className="sun-icon" aria-hidden="true">☀️</span>
          <span className="brand-title">SPEAR</span>
          <span className="brand-subtle">Earth Observation Demo</span>
        </div>
        <div className="top-right">
          <span className="brand-subtle">Plaksha University · Purdue University</span>
          <a
            className="paper-btn"
            href="https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Ranjan_SPEAR_Self-Supervised_Sample_Efficient_Pixel-Level_Multi-Modal_Spectral_Fusion_for_Earth_WACVW_2026_paper.html"
            target="_blank"
            rel="noreferrer"
          >
            Paper ↗
          </a>
        </div>
      </header>

      <div className="hero">
        <div className="hero-text">
          <div className="eyebrow">SPEAR · Earth Intelligence</div>
          <h1>Spatial QA with land cover, crop, and flood signals fused into one map.</h1>
          <p className="lede">Query India-wide pixels using districts or coordinates. We blend Sentinel-1/2, Planet, and climate streams to return English answers with inline heatmaps.</p>
          <div className="hero-actions">
            <div className="hero-chips">
              <button className="chip ghost link-chip" type="button" onClick={() => applySample(defaultQuery)}>
                Insert sample prompt
              </button>
              <span className="chip ghost">Coverage 2020–2024</span>
              <span className="chip ghost">Inline heatmaps</span>
              <span className="chip ghost">Multi-sensor fusion</span>
            </div>
          </div>
        </div>
          <div className="hero-card">
          <div className="hero-metric">
            <div className="hero-label">Sensors</div>
            <div className="hero-value">S2 · S1 · Planet · Climate</div>
          </div>
          <div className="hero-metric">
            <div className="hero-label">Query styles</div>
            <div className="hero-value">Place · District · Coords</div>
          </div>
          <div className="hero-foot">Earth observation demo</div>
        </div>
      </div>

      <div className="mod-row">
        <span className="mod-label">Modalities</span>
        {availableMods.length ? (
          availableMods.map((mod) => (
            <span key={mod} className="mod-chip">{mod.toUpperCase()}</span>
          ))
        ) : (
          <span className="muted">Fetching available modalities…</span>
        )}
      </div>

      <main className="layout">
        <div className="column">
          <section className="panel">
            <div className="panel-title">Query</div>
            <p className="panel-hint">Use place names, districts, or coordinates. Add a year or modality if you need precision.</p>

            <form className="form" onSubmit={handleSubmit}>
              <label className="input-label" htmlFor="query">Question</label>
              <textarea
                id="query"
                className="textarea"
                placeholder="Example: Show flood extent near Guwahati for 2023."
                rows={4}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />

              <div className="input-label">Choose modalities</div>
              <div className="pill-row">
                {availableMods.map((mod) => {
                  const active = selectedMods.includes(mod);
                  return (
                    <button
                      type="button"
                      key={mod}
                      className={`pill ${active ? 'pill-active' : ''}`}
                      onClick={() => toggleModality(mod)}
                    >
                      {mod.toUpperCase()}
                    </button>
                  );
                })}
                {!availableMods.length && <span className="muted">No modalities detected; backend will use all.</span>}
              </div>

              <div className="actions">
                <button className="btn btn-primary" type="submit" disabled={loading}>
                  {loading ? 'Running...' : 'Run query'}
                </button>
              </div>
            </form>
          </section>

          <section className="panel">
            <div className="panel-title">Quick prompts</div>
            <p className="panel-hint">Tap to prefill. You can still edit before running.</p>
            <div className="sample-grid">
              {samples.map((item) => (
                <button
                  key={item.title}
                  className="sample"
                  type="button"
                  onClick={() => applySample(item.text)}
                  disabled={loading}
                >
                  <div className="sample-title">{item.title}</div>
                  <div className="sample-text">{item.text}</div>
                </button>
              ))}
            </div>
          </section>
        </div>

        <div className="column">
          <section className="panel results-panel">
            <div className="panel-head-row">
              <div>
                <div className="panel-title">Response</div>
                <p className="panel-hint">Answers in English with pixel counts and modality usage.</p>
              </div>
              {error && <div className="alert alert-danger">{error}</div>}
            </div>

            <div className="stat-grid">
              <div className="stat">
                <div className="stat-label">Elapsed</div>
                <div className="stat-value">{hasResult ? `${result.elapsed_s}s` : '--'}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Pixels</div>
                <div className="stat-value">{hasResult ? result.total_pixels.toLocaleString() : '--'}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Modalities</div>
                <div className="stat-value">{selectedMods.length ? selectedMods.join(', ') : 'All available'}</div>
              </div>
            </div>

            <div className="answer-block">
              <div className="answer-head">Answer</div>
              {hasResult ? <pre className="answer-text">{result.answer}</pre> : <p className="muted">Run a query to see the generated answer.</p>}
            </div>

            {hasResult && result.query_summary && (
              <div className="summary">
                <div className="summary-head">Query summary</div>
                <p>{result.query_summary}</p>
              </div>
            )}

            {modalityBadges.length > 0 && (
              <div className="chips">
                {modalityBadges.map((item) => (
                  <span key={item.name} className="chip">
                    {item.name.toUpperCase()}: {item.count.toLocaleString()} px
                  </span>
                ))}
              </div>
            )}
          </section>

          <section className="panel map-panel">
            <div className="panel-title">Map preview</div>
            <p className="panel-hint">Heatmap is rendered inline. Hover and pan inside the frame.</p>
            {mapDoc ? (
              <iframe title="Result map" className="map-frame" srcDoc={mapDoc} sandbox="allow-scripts allow-same-origin" />
            ) : (
              <div className="map-placeholder">Results map will appear here after you run a query.</div>
            )}
          </section>
        </div>
      </main>

      <footer className="footer">
        <div className="footer-meta">© {new Date().getFullYear()} SPEAR · Spatial Earth Analytics</div>
        <div className="footer-note">The authors acknowledge CNH Industrial for field resources and collaboration support.</div>
      </footer>
    </div>
  );
}

export default App;
