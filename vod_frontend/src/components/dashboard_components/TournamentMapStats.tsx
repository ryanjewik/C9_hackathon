import { useState, useEffect } from 'react';

interface TournamentMapStat {
  tournamentName: string;
  map: string;
  count: number;
}

export function TournamentMapStats() {
  const [stats, setStats] = useState<TournamentMapStat[]>([]);

  useEffect(() => {
    async function fetchStats() {
      try {
        const response = await fetch('/dashboard/tournament_map_stats');
        if (!response.ok) {
          console.error("Dashboard API error: ", response.status, response.statusText);
          return;
        }
        const data = await response.json();
        setStats(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error("Failed to load tournament map stats: ", error);
      }
    }
    fetchStats();
  }, []);

  // Group by tournament name for display
  const byTournament: Record<string, TournamentMapStat[]> = {};
  for (const row of stats) {
    if (!byTournament[row.tournamentName]) byTournament[row.tournamentName] = [];
    byTournament[row.tournamentName].push(row);
  }

  return (
    <div className="mt-6 h-96 max-w-7xl rounded-2xl mx-auto bg-white bg-opacity-55 p-4 justify-items-center border-2 border-c9-cyan hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      <h1 className="text-2xl font-bold tracking-wide">
        <span className="text-c9-cyan font-extrabold">{"Maps"}</span>
      </h1>
      <h1>VCT 2026 Map Play Count by Tournament</h1>
      <ul>
        {Object.entries(byTournament).map(([tournament, rows]) => (
          <li key={tournament}>
            {tournament}: {rows.map(r => `${r.map} (${r.count})`).join(', ')}
          </li>
        ))}
      </ul>
    </div>
  );
}
