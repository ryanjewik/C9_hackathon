import { useState, useEffect } from 'react';

interface Match {
  id: number;
  phase: string;
  date: string;
  tournamentName: string;
  team1Name: string;
  team1Score: number;
  team1Id: number;
  team2Name: string;
  team2Score: number;
  team2Id: number;
  winner: number;
  map1: string;
  map2: string | null;
  map3: string | null;
  map4: string | null;
  map5: string | null;
}

interface MapStat {
  mapSelected: string;
  pickCount: number;
  banCount: number;
}

export function RecentMatches() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [mapStats, setMapStats] = useState<Record<number, MapStat[]>>({});

  useEffect(() => {
    async function fetchMatches() {
      try {
        const response = await fetch('/dashboard/recent_matches');
        if (!response.ok) {
          console.error("Dashboard API error: ", response.status, response.statusText);
          return;
        }
        const data: Match[] = await response.json();
        setMatches(Array.isArray(data) ? data : []);

        const teamIds = [...new Set(data.flatMap(m => [m.team1Id, m.team2Id]))];
        const statsEntries = await Promise.all(
          teamIds.map(async (teamId) => {
            const res = await fetch(`/dashboard/map_stats?teamId=${teamId}`);
            if (!res.ok) return [teamId, []] as [number, MapStat[]];
            const stats: MapStat[] = await res.json();
            return [teamId, stats] as [number, MapStat[]];
          })
        );
        setMapStats(Object.fromEntries(statsEntries));
      } catch (error) {
        console.error("Failed to load recent matches: ", error);
      }
    }
    fetchMatches();
  }, []);

  return (
    <div className="mt-6 h-96 max-w-7xl rounded-2xl mx-auto bg-white bg-opacity-55 p-4 justify-items-center border-2 border-c9-cyan hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      <h1 className="text-2xl font-bold tracking-wide">
        <span className="text-c9-cyan font-extrabold">{"Recent"}</span>
      </h1>
      <h1>Recent VCT Matches</h1>
      <ul>
        {matches.map((m) => (
          <li key={m.id}>
            {new Date(m.date).toLocaleDateString()} - {m.team1Name} {m.team1Score} - {m.team2Score} {m.team2Name} ({m.tournamentName})
            <ul>
              <li>{m.team1Name} maps: {(mapStats[m.team1Id] ?? []).map(s => `${s.mapSelected} Pick:${s.pickCount} Ban:${s.banCount}`).join(', ')}</li>
              <li>{m.team2Name} maps: {(mapStats[m.team2Id] ?? []).map(s => `${s.mapSelected} Pick:${s.pickCount} Ban:${s.banCount}`).join(', ')}</li>
            </ul>
          </li>
        ))}
      </ul>
    </div>
  );
}
