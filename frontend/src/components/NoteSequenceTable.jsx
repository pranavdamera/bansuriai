/**
 * NoteSequenceTable
 *
 * Renders each decoded segment as a table row showing:
 * note label, start time, end time, duration, and confidence.
 * Confidence cells are color-coded: green >= 85%, yellow >= 65%, red below.
 */
export default function NoteSequenceTable({ sequence }) {
  if (!sequence || sequence.length === 0) return null;

  return (
    <div className="card card-glow overflow-hidden animate-slide-up" style={{ animationDelay: '0.1s' }}>
      <div className="border-b border-clay-800/30 px-6 py-4">
        <h2 className="font-display text-lg font-bold text-clay-100">Note Sequence</h2>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-clay-800/20 text-xs uppercase tracking-wider text-clay-600">
              <th className="px-6 py-3 text-left">#</th>
              <th className="px-6 py-3 text-left">Swara</th>
              <th className="px-6 py-3 text-right">Start</th>
              <th className="px-6 py-3 text-right">End</th>
              <th className="px-6 py-3 text-right">Duration</th>
              <th className="px-6 py-3 text-right">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {sequence.map((seg, i) => {
              const dur = (seg.end - seg.start).toFixed(2);
              const confPct = Math.round(seg.confidence * 100);
              const confClass = getConfClass(seg.confidence);

              return (
                <tr
                  key={i}
                  className="border-b border-clay-800/10 transition-colors hover:bg-clay-900/30"
                >
                  <td className="px-6 py-3 text-clay-600">{i + 1}</td>
                  <td className="px-6 py-3">
                    <span className="inline-flex h-8 w-12 items-center justify-center rounded-md bg-saffron-500/10 font-display text-base font-bold text-saffron-400 ring-1 ring-saffron-500/20">
                      {seg.note}
                    </span>
                  </td>
                  <td className="px-6 py-3 text-right font-mono text-clay-400">
                    {seg.start.toFixed(2)}s
                  </td>
                  <td className="px-6 py-3 text-right font-mono text-clay-400">
                    {seg.end.toFixed(2)}s
                  </td>
                  <td className="px-6 py-3 text-right font-mono text-clay-400">
                    {dur}s
                  </td>
                  <td className="px-6 py-3 text-right">
                    <span className={`font-semibold ${confClass}`}>{confPct}%</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function getConfClass(conf) {
  if (conf >= 0.85) return 'confidence-high';
  if (conf >= 0.65) return 'confidence-medium';
  return 'confidence-low';
}
