/**
 * TimelineChart
 *
 * Horizontal timeline showing each note as a colored block positioned
 * by start/end time along a time axis. Block opacity reflects confidence.
 * Built with plain divs rather than Recharts for precise control over
 * the horizontal bar layout.
 */
export default function TimelineChart({ sequence }) {
  if (!sequence || sequence.length === 0) return null;

  const totalEnd = Math.max(...sequence.map((s) => s.end), 0.1);

  // Assign each unique swara a consistent color
  const SWARA_COLORS = {
    Sa:  { bg: 'bg-amber-500',   ring: 'ring-amber-400/30' },
    Re:  { bg: 'bg-orange-500',  ring: 'ring-orange-400/30' },
    Ga:  { bg: 'bg-yellow-500',  ring: 'ring-yellow-400/30' },
    Ma:  { bg: 'bg-lime-500',    ring: 'ring-lime-400/30' },
    Pa:  { bg: 'bg-emerald-500', ring: 'ring-emerald-400/30' },
    Dha: { bg: 'bg-teal-500',    ring: 'ring-teal-400/30' },
    Ni:  { bg: 'bg-cyan-500',    ring: 'ring-cyan-400/30' },
  };

  const fallback = { bg: 'bg-clay-600', ring: 'ring-clay-500/30' };

  return (
    <div className="card card-glow p-6 animate-slide-up" style={{ animationDelay: '0.2s' }}>
      <h2 className="mb-4 font-display text-lg font-bold text-clay-100">Note Timeline</h2>

      {/* Timeline bar */}
      <div className="relative h-14 rounded-lg bg-clay-900/50 ring-1 ring-clay-800/20">
        {sequence.map((seg, i) => {
          const leftPct = (seg.start / totalEnd) * 100;
          const widthPct = ((seg.end - seg.start) / totalEnd) * 100;
          const colors = SWARA_COLORS[seg.note] || fallback;
          const opacity = 0.5 + seg.confidence * 0.5;  // 50–100% opacity

          return (
            <div
              key={i}
              title={`${seg.note}: ${seg.start.toFixed(2)}s – ${seg.end.toFixed(2)}s (${Math.round(seg.confidence * 100)}%)`}
              className={`absolute top-2 bottom-2 flex items-center justify-center rounded-md ${colors.bg} ring-1 ${colors.ring} transition-all`}
              style={{
                left: `${leftPct}%`,
                width: `${Math.max(widthPct, 2)}%`,  // Min 2% so tiny segments are visible
                opacity,
              }}
            >
              <span className="text-xs font-bold text-ink-900 drop-shadow-sm">
                {widthPct > 6 ? seg.note : ''}
              </span>
            </div>
          );
        })}
      </div>

      {/* Time axis labels */}
      <div className="mt-2 flex justify-between px-1 text-xs font-mono text-clay-600">
        <span>0.00s</span>
        <span>{(totalEnd / 2).toFixed(2)}s</span>
        <span>{totalEnd.toFixed(2)}s</span>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-3">
        {sequence
          .filter((seg, i, arr) => arr.findIndex((s) => s.note === seg.note) === i)
          .map((seg) => {
            const colors = SWARA_COLORS[seg.note] || fallback;
            return (
              <div key={seg.note} className="flex items-center gap-1.5 text-xs text-clay-400">
                <span className={`h-2.5 w-2.5 rounded-sm ${colors.bg}`} />
                {seg.note}
              </div>
            );
          })}
      </div>
    </div>
  );
}
