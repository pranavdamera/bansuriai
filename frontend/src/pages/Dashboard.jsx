import { useState, useCallback, useEffect, useRef } from 'react';
import { analyzeAudio, checkHealth } from '../services/api';
import AudioUploader from '../components/AudioUploader';
import AnalyzeButton from '../components/AnalyzeButton';
import ErrorBanner from '../components/ErrorBanner';
import SummaryCard from '../components/SummaryCard';
import NoteSequenceTable from '../components/NoteSequenceTable';
import ConfidenceChart from '../components/ConfidenceChart';
import TimelineChart from '../components/TimelineChart';
import LiveModePanel from '../components/LiveModePanel';

export default function Dashboard() {
  // ── Backend connection state ────────────────────────────────────────
  const [backendStatus, setBackendStatus] = useState('checking');
  const [modelLoaded, setModelLoaded] = useState(false);
  const retryTimer = useRef(null);

  // ── Tab state ──────────────────────────────────────────────────────
  // "upload" = batch file analysis, "live" = real-time mic mode
  const [activeTab, setActiveTab] = useState('upload');

  // ── Analysis state (upload tab) ────────────────────────────────────
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // ── Health check on mount + periodic retry ─────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const health = await checkHealth();
        if (!cancelled) {
          setBackendStatus('connected');
          setModelLoaded(health.model_loaded);
          if (retryTimer.current) {
            clearInterval(retryTimer.current);
            retryTimer.current = null;
          }
        }
      } catch {
        if (!cancelled) {
          setBackendStatus('disconnected');
          if (!retryTimer.current) {
            retryTimer.current = setInterval(check, 5000);
          }
        }
      }
    }

    check();

    return () => {
      cancelled = true;
      if (retryTimer.current) clearInterval(retryTimer.current);
    };
  }, []);

  // ── Handlers ───────────────────────────────────────────────────────
  const handleFileSelect = useCallback((selectedFile) => {
    setFile(selectedFile);
    setError(null);
    setResult(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeAudio(file);
      setResult(response);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  }, [file]);

  const handleDismissError = useCallback(() => setError(null), []);

  const canAnalyze = file && backendStatus === 'connected' && !loading;

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="mb-10 text-center">
        <h1 className="font-display text-4xl font-bold tracking-tight text-clay-100 sm:text-5xl">
          Bansuri<span className="text-saffron-400">AI</span>
        </h1>
        <p className="mt-2 text-base text-clay-600">
          Real-time pitch classification and feedback for bansuri practice.
        </p>
      </header>

      {/* ── Backend status banner ──────────────────────────────────── */}
      {backendStatus === 'disconnected' && (
        <div className="mb-6 animate-slide-up flex items-center gap-3 rounded-xl border border-amber-500/20 bg-amber-950/30 px-5 py-4">
          <span className="relative flex h-3 w-3 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
            <span className="relative inline-flex h-3 w-3 rounded-full bg-amber-500" />
          </span>
          <div className="flex-1">
            <p className="text-sm font-medium text-amber-200">Backend not reachable</p>
            <p className="text-xs text-amber-200/60">
              Start the server: <code className="font-mono text-amber-300">cd backend &amp;&amp; python run.py</code>
              <span className="ml-2 text-amber-200/40">— retrying automatically…</span>
            </p>
          </div>
        </div>
      )}

      {backendStatus === 'connected' && !modelLoaded && (
        <div className="mb-6 animate-slide-up flex items-center gap-3 rounded-xl border border-saffron-500/20 bg-saffron-950/20 px-5 py-4">
          <span className="flex h-3 w-3 shrink-0 rounded-full bg-saffron-500" />
          <p className="text-sm text-saffron-200/80">
            Server running in <span className="font-semibold text-saffron-300">placeholder mode</span>
            — no trained model loaded. Predictions will be random.
          </p>
        </div>
      )}

      {backendStatus === 'checking' && (
        <div className="mb-6 flex items-center justify-center gap-2 py-3 text-sm text-clay-600">
          <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Connecting to backend…
        </div>
      )}

      {/* ── Mode tabs ──────────────────────────────────────────────── */}
      <div className="mb-8 flex rounded-xl border border-clay-800/30 bg-clay-900/40 p-1 gap-1">
        <button
          onClick={() => setActiveTab('upload')}
          className={`flex-1 rounded-lg px-4 py-2.5 text-sm font-medium transition-all ${
            activeTab === 'upload'
              ? 'bg-clay-800/70 text-clay-100 shadow-sm'
              : 'text-clay-600 hover:text-clay-400'
          }`}
        >
          Upload &amp; Analyze
        </button>
        <button
          onClick={() => setActiveTab('live')}
          className={`flex-1 rounded-lg px-4 py-2.5 text-sm font-medium transition-all ${
            activeTab === 'live'
              ? 'bg-clay-800/70 text-clay-100 shadow-sm'
              : 'text-clay-600 hover:text-clay-400'
          }`}
        >
          <span className="flex items-center justify-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                activeTab === 'live' ? 'bg-red-400 animate-pulse' : 'bg-clay-600'
              }`}
            />
            Live Mode
          </span>
        </button>
      </div>

      {/* ── Upload tab ─────────────────────────────────────────────── */}
      {activeTab === 'upload' && (
        <>
          <section className="mb-8 space-y-4">
            <AudioUploader onFileSelect={handleFileSelect} disabled={loading} />
            <AnalyzeButton
              onAnalyze={handleAnalyze}
              loading={loading}
              disabled={!canAnalyze}
            />
          </section>

          {error && (
            <section className="mb-8">
              <ErrorBanner message={error} onDismiss={handleDismissError} />
            </section>
          )}

          {loading && (
            <section className="py-16 text-center animate-fade-in">
              <div className="mx-auto mb-4 h-10 w-10 animate-pulse-warm rounded-full bg-saffron-500/20 ring-2 ring-saffron-500/40" />
              <p className="text-sm text-clay-600">
                Processing audio through the recognition pipeline…
              </p>
              <p className="mt-1 text-xs text-clay-700">
                Preprocessing → Feature extraction → Model inference → Decoding
              </p>
            </section>
          )}

          {result && !loading && (
            <section className="space-y-6">
              <div className="flex items-center gap-4">
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-clay-700/40 to-transparent" />
                <span className="text-xs font-medium uppercase tracking-widest text-clay-600">
                  Analysis Results
                </span>
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-clay-700/40 to-transparent" />
              </div>

              <SummaryCard result={result} />
              <TimelineChart sequence={result.decoded_sequence} />

              <div className="grid gap-6 lg:grid-cols-2">
                <ConfidenceChart sequence={result.decoded_sequence} />
                <NoteSequenceTable sequence={result.decoded_sequence} />
              </div>
            </section>
          )}
        </>
      )}

      {/* ── Live tab ───────────────────────────────────────────────── */}
      {activeTab === 'live' && (
        <LiveModePanel disabled={backendStatus !== 'connected'} />
      )}

      {/* ── Footer ─────────────────────────────────────────────────── */}
      <footer className="mt-16 border-t border-clay-800/20 pt-6 text-center text-xs text-clay-700">
        BansuriAI — Real-time bansuri pitch tutor
        <span className="mx-2 text-clay-800">·</span>
        {backendStatus === 'connected' && (
          <span className="text-emerald-600">
            Backend connected {modelLoaded ? '· Model loaded' : '· Placeholder mode'}
          </span>
        )}
        {backendStatus === 'disconnected' && (
          <span className="text-amber-600">Backend disconnected</span>
        )}
      </footer>
    </div>
  );
}
