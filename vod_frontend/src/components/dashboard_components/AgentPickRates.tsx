import { useState, useEffect, useMemo } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

interface AgentPickRate {
  tournamentName: string;
  agent: string;
  agentPicks: number;
  totalMatches: number;
  pickRate: number;
}

// 30 visually distinct colors across blues, purples, teals, greens, warm accents
const SLICE_COLORS = [
  '#4dd9e8', '#9b59b6', '#1abc9c', '#e74c3c', '#2a9db5',
  '#7d3c98', '#16a085', '#e67e22', '#5bbfe8', '#bb8fce',
  '#52be80', '#f1c40f', '#1a7a9a', '#6c3483', '#76d7c4',
  '#e91e8c', '#36b8d4', '#d2b4de', '#0e8a75', '#ff6b9d',
  '#0e5c78', '#a569bd', '#1e8449', '#f39c12', '#63cfdf',
  '#884ea0', '#148f77', '#cb4335', '#2e86c1', '#a9cce3',
];

const RADIAN = Math.PI / 180;
const MIN_PCT_FOR_LABEL = 0.04; // only draw pointer labels for slices >= 4%

interface PieEntry { agent: string; pickRate: number; agentPicks: number }

interface LabelProps {
  cx?: number; cy?: number; midAngle?: number;
  outerRadius?: number; percent?: number; index: number;
  pieData: PieEntry[];
}

function PointerLabel({ cx, cy, midAngle, outerRadius, percent, index, pieData }: LabelProps) {
  if (!cx || !cy || midAngle == null || !outerRadius || !percent) return null;
  if (percent < MIN_PCT_FOR_LABEL) return null;

  const entry = pieData[index];
  const color = SLICE_COLORS[index % SLICE_COLORS.length];
  const ICON_SIZE = 12;
  const GAP = 5;
  const SPOKE = 13;
  const FLAT = 9;

  const sx = cx + (outerRadius + GAP) * Math.cos(-midAngle * RADIAN);
  const sy = cy + (outerRadius + GAP) * Math.sin(-midAngle * RADIAN);
  const mx = cx + (outerRadius + GAP + SPOKE) * Math.cos(-midAngle * RADIAN);
  const my = cy + (outerRadius + GAP + SPOKE) * Math.sin(-midAngle * RADIAN);
  const isLeft = mx < cx;
  const ex = mx + (isLeft ? -FLAT : FLAT);
  const ey = my;

  // icon sits just outside the elbow end
  const iconX = isLeft ? ex - FLAT - ICON_SIZE - 2 : ex + 2;
  const textX = isLeft ? iconX - 3 : iconX + ICON_SIZE + 3;
  const textAnchor = isLeft ? 'end' : 'start';

  return (
    <g>
      <polyline
        points={`${sx},${sy} ${mx},${my} ${ex},${ey}`}
        fill="none"
        stroke={color}
        strokeWidth={1.2}
        opacity={0.85}
      />
      <circle cx={ex} cy={ey} r={2} fill={color} />
      <image
        href={`/agents/${entry.agent.toLowerCase().trim()}.webp`}
        x={iconX}
        y={ey - ICON_SIZE / 2}
        width={ICON_SIZE}
        height={ICON_SIZE}
        style={{ borderRadius: 8 }}
      />
      <text
        x={textX}
        y={ey}
        textAnchor={textAnchor}
        fill="#1a2a3a"
        fontSize={9}
        fontWeight={600}
        dominantBaseline="middle"
      >
        {(percent * 100).toFixed(0)}%
      </text>
    </g>
  );
}

interface TooltipProps {
  active?: boolean;
  payload?: { value: number; payload: PieEntry }[];
}
function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const { agent, agentPicks } = payload[0].payload;
  const pct = (payload[0].value * 100).toFixed(1);
  return (
    <div className="bg-white border border-c9-cyan rounded-lg px-3 py-2 shadow text-sm text-c9-text">
      <div className="flex items-center gap-1.5 font-semibold mb-0.5">
        <img
          src={`/agents/${agent.toLowerCase().trim()}.webp`}
          alt={agent}
          width={18}
          height={18}
          className="rounded-full object-cover"
          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
        />
        <span className="capitalize">{agent}</span>
      </div>
      <div>Pick rate: <span className="text-c9-cyan font-bold">{pct}%</span></div>
      <div className="text-c9-muted">Picks: {agentPicks}</div>
    </div>
  );
}

export function AgentPickRates() {
  const [stats, setStats] = useState<AgentPickRate[]>([]);
  const [selectedTournament, setSelectedTournament] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStart = Date.now();
    async function fetchStats() {
      try {
        const response = await fetch('/dashboard/agent_pickrates');
        if (!response.ok) return;
        const data = await response.json();
        const rows: AgentPickRate[] = Array.isArray(data) ? data : [];
        setStats(rows);
        if (rows.length > 0) setSelectedTournament(rows[0].tournamentName);
      } catch (error) {
        console.error('Failed to load agent pick rates:', error);
      } finally {
        const elapsed = Date.now() - fetchStart;
        setTimeout(() => setLoading(false), Math.max(0, 650 - elapsed));
      }
    }
    fetchStats();
  }, []);

  const tournaments = useMemo(
    () => [...new Set(stats.map((r) => r.tournamentName))],
    [stats],
  );

  const pieData: PieEntry[] = useMemo(() =>
    stats
      .filter((r) => r.tournamentName === selectedTournament)
      .sort((a, b) => b.pickRate - a.pickRate)
      .map((r) => ({ agent: r.agent, pickRate: r.pickRate, agentPicks: r.agentPicks })),
    [stats, selectedTournament],
  );

  return (
    <div className="h-full rounded-2xl bg-white bg-opacity-55 p-4 border-2 border-c9-cyan hover:shadow-lg hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <h2 className="text-2xl font-bold tracking-wide">
          <span className="text-c9-cyan font-extrabold">Agent Pick Rates</span>
        </h2>
        {!loading && (
          <select
            value={selectedTournament}
            onChange={(e) => setSelectedTournament(e.target.value)}
            className="text-sm border border-c9-cyan rounded-lg px-2 py-1 bg-white text-c9-text focus:outline-none focus:ring-2 focus:ring-c9-cyan"
          >
            {tournaments.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center animate-pulse" style={{ height: 280 }}>
          <div className="w-32 h-32 rounded-full border-8 border-c9-cyan/20" />
        </div>
      ) : (
        <>
        <ResponsiveContainer width="100%" height={280}>
        <PieChart margin={{ top: 10, right: 65, bottom: 10, left: 65 }}>
          <Pie
            data={pieData}
            dataKey="pickRate"
            nameKey="agent"
            cx="50%"
            cy="50%"
            outerRadius={60}
            innerRadius={27}
            paddingAngle={1}
            labelLine={false}
            label={(props) => (
              <PointerLabel {...props} pieData={pieData} />
            )}
          >
            {pieData.map((entry, i) => (
              <Cell key={entry.agent} fill={SLICE_COLORS[i % SLICE_COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>

      {/* Legend grid: color dot + icon + name + % for every agent */}
      <div className="flex flex-wrap justify-center gap-x-5 gap-y-2 mt-1 px-2">
        {pieData.map((entry, i) => (
          <div key={entry.agent} className="flex items-center gap-1 text-xs text-c9-text">
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: SLICE_COLORS[i % SLICE_COLORS.length] }}
            />
            <img
              src={`/agents/${entry.agent.toLowerCase().trim()}.webp`}
              alt={entry.agent}
              width={15}
              height={15}
              className="rounded-full object-cover"
              onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
            />
            <span className="capitalize">{entry.agent}</span>
            <span className="text-c9-muted">{(entry.pickRate * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
      </>
      )}
    </div>
  );
}
