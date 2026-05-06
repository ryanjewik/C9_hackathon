import { useState, useEffect } from 'react';

interface AgentPickRate {
  tournamentName: string;
  agent: string;
  agentPicks: number;
  totalMatches: number;
  pickRate: number;
}

export function AgentPickRates() {
  const [stats, setStats] = useState<AgentPickRate[]>([]);

  useEffect(() => {
    async function fetchStats() {
      try {
        const response = await fetch('/dashboard/agent_pickrates');
        if (!response.ok) {
          console.error("Dashboard API error: ", response.status, response.statusText);
          return;
        }
        const data = await response.json();
        setStats(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error("Failed to load agent pick rates: ", error);
      }
    }
    fetchStats();
  }, []);

  // Group by tournament name for display
  const byTournament: Record<string, AgentPickRate[]> = {};
  for (const row of stats) {
    if (!byTournament[row.tournamentName]) byTournament[row.tournamentName] = [];
    byTournament[row.tournamentName].push(row);
  }

  return (
    <div className="mt-6 h-96 max-w-7xl rounded-2xl mx-auto bg-white bg-opacity-55 p-4 justify-items-center border-2 border-c9-cyan hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      <h1 className="text-2xl font-bold tracking-wide">
        <span className="text-c9-cyan font-extrabold">{"Agent Pick Rates"}</span>
      </h1>
      <h1>VCT 2026 Agent Pick Rates by Tournament</h1>
      <ul>
        {Object.entries(byTournament).map(([tournament, rows]) => (
          <li key={tournament}>
            {tournament}: {rows.map(r => `${r.agent} ${(r.pickRate * 100).toFixed(1)}%`).join(', ')}
          </li>
        ))}
      </ul>
    </div>
  );
}
