import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ResponsiveContainer, LabelList,
} from 'recharts';

interface TournamentMapStat {
  tournamentName: string;
  map: string;
  count: number;
}

// One distinct color per map (stable, not tournament-dependent)
const MAP_COLORS: Record<string, string> = {
  Abyss:    '#4dd9e8',
  Ascent:   '#9b59b6',
  Bind:     '#e74c3c',
  Breeze:   '#1abc9c',
  Corrode:  '#e67e22',
  Fracture: '#2e86c1',
  Haven:    '#f1c40f',
  Lotus:    '#e91e8c',
  Pearl:    '#52be80',
  Split:    '#bb8fce',
};
const DEFAULT_COLOR = '#4dd9e8';

interface TooltipProps {
  active?: boolean;
  payload?: { value: number; payload: { map: string; count: number } }[];
}
function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const { map, count } = payload[0].payload;
  return (
    <div className="bg-white border border-c9-cyan rounded-lg px-3 py-2 shadow text-sm text-c9-text">
      <div className="font-semibold mb-0.5" style={{ color: MAP_COLORS[map] ?? DEFAULT_COLOR }}>
        {map}
      </div>
      <div>Times played: <span className="font-bold">{count}</span></div>
    </div>
  );
}

export function TournamentMapStats() {
  const [stats, setStats] = useState<TournamentMapStat[]>([]);
  const [selectedTournament, setSelectedTournament] = useState<string>('');

  useEffect(() => {
    async function fetchStats() {
      try {
        const response = await fetch('/dashboard/tournament_map_stats');
        if (!response.ok) return;
        const data = await response.json();
        const rows: TournamentMapStat[] = Array.isArray(data) ? data : [];
        setStats(rows);
        if (rows.length > 0) setSelectedTournament(rows[0].tournamentName);
      } catch (error) {
        console.error('Failed to load tournament map stats:', error);
      }
    }
    fetchStats();
  }, []);

  const tournaments = useMemo(
    () => [...new Set(stats.map((r) => r.tournamentName))],
    [stats],
  );

  const chartData = useMemo(() =>
    stats
      .filter((r) => r.tournamentName === selectedTournament)
      .sort((a, b) => b.count - a.count),
    [stats, selectedTournament],
  );

  const maxCount = useMemo(
    () => Math.max(...chartData.map((r) => r.count), 0),
    [chartData],
  );

  return (
    <div className="h-full rounded-2xl bg-white bg-opacity-55 p-4 border-2 border-c9-cyan hover:shadow-lg hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <h2 className="text-2xl font-bold tracking-wide">
          <span className="text-c9-cyan font-extrabold">Map Pick Rates</span>
        </h2>
        <select
          value={selectedTournament}
          onChange={(e) => setSelectedTournament(e.target.value)}
          className="text-sm border border-c9-cyan rounded-lg px-2 py-1 bg-white text-c9-text focus:outline-none focus:ring-2 focus:ring-c9-cyan"
        >
          {tournaments.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 4, right: 56, bottom: 4, left: 8 }}
          barCategoryGap="28%"
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#d0e8f0" />
          <XAxis
            type="number"
            domain={[0, maxCount + 2]}
            tick={{ fontSize: 11, fill: '#6b8ca8' }}
            tickLine={false}
            axisLine={{ stroke: '#d0e8f0' }}
          />
          <YAxis
            type="category"
            dataKey="map"
            width={68}
            tick={{ fontSize: 13, fontWeight: 600, fill: '#1a2a3a' }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(77,217,232,0.08)' }} />
          <Bar dataKey="count" radius={[0, 6, 6, 0]}>
            {chartData.map((entry) => (
              <Cell key={entry.map} fill={MAP_COLORS[entry.map] ?? DEFAULT_COLOR} />
            ))}
            <LabelList
              dataKey="count"
              position="right"
              style={{ fontSize: 12, fontWeight: 700, fill: '#1a2a3a' }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
