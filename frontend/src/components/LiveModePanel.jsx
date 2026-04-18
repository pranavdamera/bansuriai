import { useState, useRef, useCallback, useEffect } from 'react';
import { classifyAudio } from '../services/api';

// How many Float32 samples to accumulate before sending a classify request.
// 22050 * 1.5 ≈ 1.49 seconds — minimum the model needs for a full inference window.
const TARGET_SAMPLES = Math.floor(22050 * 1.5);
const TARGET_SAMPLE_RATE = 22050;

// Intonation needle range in cents
const CENTS_RANGE = 50;

export default function LiveModePanel({ disabled }) {
  const [isLive, setIsLive] = useState(false);
  const [micError, setMicError] = useState(null);
  const [result, setResult] = useState(null);      // latest ClassifyResponse
  const [pending, setPending] = useState(false);   // request in-flight

  const audioCtxRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const bufferRef = useRef([]);
  const sendingRef = useRef(false);  // prevent overlapping requests

  // ── Start live capture ──────────────────────────────────────────────
  const startLive = useCallback(async () => {
    setMicError(null);
    setResult(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: TARGET_SAMPLE_RATE },
        video: false,
      });

      // Request target sample rate; browser may grant a different rate.
      const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      const source = ctx.createMediaStreamSource(stream);

      // ScriptProcessorNode accumulates samples into our buffer.
      // bufferSize=4096 keeps latency low while giving stable callbacks.
      const processor = ctx.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        const chunk = e.inputBuffer.getChannelData(0);
        bufferRef.current.push(...chunk);

        if (bufferRef.current.length >= TARGET_SAMPLES && !sendingRef.current) {
          sendingRef.current = true;
          const pcm = new Float32Array(bufferRef.current.splice(0, TARGET_SAMPLES));
          const actualRate = ctx.sampleRate;

          setPending(true);
          classifyAudio(pcm, actualRate)
            .then((res) => setResult(res))
            .catch((err) => console.warn('classify error:', err))
            .finally(() => {
              setPending(false);
              sendingRef.current = false;
            });
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      audioCtxRef.current = ctx;
      processorRef.current = processor;
      streamRef.current = stream;
      bufferRef.current = [];
      setIsLive(true);
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setMicError('Microphone access denied. Allow microphone permission and try again.');
      } else if (err.name === 'NotFoundError') {
        setMicError('No microphone found. Connect a microphone and try again.');
      } else {
        setMicError(`Could not access microphone: ${err.message}`);
      }
    }
  }, []);

  // ── Stop live capture ───────────────────────────────────────────────
  const stopLive = useCallback(() => {
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    processorRef.current = null;
    streamRef.current = null;
    audioCtxRef.current = null;
    bufferRef.current = [];
    setIsLive(false);
    setPending(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => () => stopLive(), [stopLive]);

  // ── Derived display values ──────────────────────────────────────────
  const note = result?.predicted_note ?? '—';
  const confidence = result ? Math.round(result.confidence * 100) : null;
  const intonation = result?.intonation ?? 'in_tune';
  const cents = result?.cents_off ?? 0;
  const feedback = result?.feedback ?? null;

  // Needle position: 0% = far left (−50¢), 50% = center (0¢), 100% = far right (+50¢)
  const needlePercent = 50 + Math.max(-CENTS_RANGE, Math.min(CENTS_RANGE, cents)) / CENTS_RANGE * 50;

  const intonationColor =
    intonation === 'in_tune' ? 'text-emerald-400' :
    intonation === 'sharp'   ? 'text-amber-400' :
                               'text-sky-400';

  const needleColor =
    intonation === 'in_tune' ? 'bg-emerald-400' :
    intonation === 'sharp'   ? 'bg-amber-400' :
                               'bg-sky-400';

  const intonationLabel =
    intonation === 'in_tune' ? 'In Tune' :
    intonation === 'sharp'   ? `Sharp +${Math.abs(cents)}¢` :
                               `Flat −${Math.abs(cents)}¢`;

  return (
    <div className="rounded-2xl border border-clay-800/30 bg-clay-900/50 p-6 space-y-6">
      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-widest text-clay-500">
            Live Mode
          </h2>
          <p className="text-xs text-clay-700 mt-0.5">
            Real-time note detection from your microphone
          </p>
        </div>

        {isLive ? (
          <button
            onClick={stopLive}
            className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-950/30 px-4 py-2 text-sm font-medium text-red-300 transition hover:bg-red-950/50 hover:text-red-200"
          >
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
            </span>
            Stop
          </button>
        ) : (
          <button
            onClick={startLive}
            disabled={disabled}
            className="flex items-center gap-2 rounded-lg border border-saffron-500/30 bg-saffron-950/20 px-4 py-2 text-sm font-medium text-saffron-300 transition hover:bg-saffron-950/40 hover:text-saffron-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" />
            </svg>
            Go Live
          </button>
        )}
      </div>

      {/* ── Mic error ────────────────────────────────────────────────── */}
      {micError && (
        <div className="rounded-lg border border-red-500/20 bg-red-950/20 px-4 py-3 text-sm text-red-300">
          {micError}
        </div>
      )}

      {/* ── Idle state ───────────────────────────────────────────────── */}
      {!isLive && !micError && (
        <div className="py-8 text-center text-clay-700 text-sm">
          Press <span className="text-saffron-400">Go Live</span> to start listening to your bansuri.
        </div>
      )}

      {/* ── Live display ─────────────────────────────────────────────── */}
      {isLive && (
        <div className="space-y-6">
          {/* Big note + confidence */}
          <div className="flex flex-col items-center gap-1 py-4">
            <div
              className={`font-display text-8xl font-bold tracking-tight transition-all duration-200 ${
                pending ? 'opacity-50' : 'opacity-100'
              } ${note === '—' ? 'text-clay-700' : 'text-clay-100'}`}
            >
              {note}
            </div>
            {confidence !== null && (
              <div className="text-sm text-clay-600">
                {confidence}% confident
                {pending && (
                  <span className="ml-2 inline-block h-1.5 w-1.5 animate-ping rounded-full bg-saffron-500" />
                )}
              </div>
            )}
            {!result && (
              <div className="mt-2 text-xs text-clay-700 animate-pulse">
                Listening… play a note
              </div>
            )}
          </div>

          {/* Intonation meter */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-clay-600">
              <span>Flat</span>
              <span className={`font-semibold ${intonationColor}`}>{intonationLabel}</span>
              <span>Sharp</span>
            </div>

            {/* Track */}
            <div className="relative h-3 rounded-full bg-clay-800/60">
              {/* Center marker */}
              <div className="absolute left-1/2 top-0 h-full w-px -translate-x-px bg-clay-600/50" />

              {/* Needle */}
              <div
                className={`absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full shadow-lg transition-all duration-300 ${needleColor}`}
                style={{ left: `${needlePercent}%` }}
              />
            </div>

            {/* Scale labels */}
            <div className="flex justify-between text-[10px] text-clay-700 px-1">
              <span>−{CENTS_RANGE}¢</span>
              <span>0</span>
              <span>+{CENTS_RANGE}¢</span>
            </div>
          </div>

          {/* Feedback */}
          {feedback && (
            <div className="rounded-xl border border-clay-800/20 bg-clay-950/40 px-4 py-3 text-sm text-clay-500 text-center">
              {feedback}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
