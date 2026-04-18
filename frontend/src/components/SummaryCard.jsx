/**
 * SummaryCard
 *
 * Top-level result display: overall confidence gauge, signal quality,
 * the summary sentence, and feedback observations.
 */
export default function SummaryCard({ result }) {
  const { overall_confidence, signal_quality_score, summary_report, feedback, detected_notes } = result;

  const confPct = Math.round(overall_confidence * 100);
  const qualPct = Math.round(signal_quality_score * 100);

  return (
    <div className="card card-glow p-6 animate-slide-up" style={{ animationDelay: '0.05s' }}>
      {/* Header row with note pills */}
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <h2 className="mr-2 font-display text-xl font-bold text-clay-100">Detected Swaras</h2>
        {detected_notes.map((note, i) => (
          <span
            key={i}
            className="rounded-full bg-saffron-500/15 px-3 py-1 text-sm font-semibold text-saffron-400 ring-1 ring-saffron-500/30"
          >
            {note}
          </span>
        ))}
      </div>

      {/* Two gauges side by side */}
      <div className="mb-5 grid grid-cols-2 gap-4">
        <GaugeStat label="Model Confidence" value={confPct} color={confColor(overall_confidence)} />
        <GaugeStat label="Signal Quality" value={qualPct} color={qualColor(signal_quality_score)} />
      </div>

      {/* Summary text */}
      <p className="mb-4 rounded-lg bg-clay-900/40 px-4 py-3 text-sm leading-relaxed text-clay-200 ring-1 ring-clay-800/30">
        {summary_report}
      </p>

      {/* Feedback list */}
      {feedback.length > 0 && (
        <ul className="space-y-1.5">
          {feedback.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-clay-400">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-saffron-500/60" />
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}


/* ── Gauge sub-component ───────────────────────────────────────────── */

function GaugeStat({ label, value, color }) {
  return (
    <div className="rounded-lg bg-clay-900/30 p-4 ring-1 ring-clay-800/20">
      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-clay-600">{label}</p>
      <div className="flex items-end gap-2">
        <span className={`font-display text-3xl font-bold ${color}`}>{value}</span>
        <span className="mb-1 text-sm text-clay-600">%</span>
      </div>
      {/* Progress bar */}
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-clay-800/50">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColor(value)}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function confColor(v) {
  if (v >= 0.85) return 'text-emerald-400';
  if (v >= 0.65) return 'text-saffron-400';
  return 'text-red-400';
}

function qualColor(v) {
  if (v >= 0.8) return 'text-emerald-400';
  if (v >= 0.5) return 'text-saffron-400';
  return 'text-red-400';
}

function barColor(pct) {
  if (pct >= 85) return 'bg-emerald-500';
  if (pct >= 65) return 'bg-saffron-500';
  return 'bg-red-500';
}
