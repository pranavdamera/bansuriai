import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

/**
 * ConfidenceChart
 *
 * Vertical bar chart with one bar per decoded segment.
 * Bar height = confidence (0–100%). Color-coded by confidence level.
 * X-axis labels show the swara name and segment index.
 */
export default function ConfidenceChart({ sequence }) {
  if (!sequence || sequence.length === 0) return null;

  const data = sequence.map((seg, i) => ({
    name: `${seg.note} (${i + 1})`,
    confidence: Math.round(seg.confidence * 100),
    raw: seg.confidence,
  }));

  return (
    <div className="card card-glow p-6 animate-slide-up" style={{ animationDelay: '0.15s' }}>
      <h2 className="mb-4 font-display text-lg font-bold text-clay-100">Confidence per Segment</h2>

      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#3e302a" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 11, fill: '#8a6e50' }}
              axisLine={{ stroke: '#3e302a' }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 11, fill: '#8a6e50' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              contentStyle={{
                background: '#1e1b18',
                border: '1px solid #3e302a',
                borderRadius: '8px',
                fontSize: '13px',
                color: '#e0cfbc',
              }}
              formatter={(value) => [`${value}%`, 'Confidence']}
            />
            <Bar dataKey="confidence" radius={[4, 4, 0, 0]} maxBarSize={48}>
              {data.map((entry, i) => (
                <Cell key={i} fill={barFill(entry.raw)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function barFill(conf) {
  if (conf >= 0.85) return '#34d399';  // emerald-400
  if (conf >= 0.65) return '#d4910e';  // saffron-500
  return '#f87171';                     // red-400
}
