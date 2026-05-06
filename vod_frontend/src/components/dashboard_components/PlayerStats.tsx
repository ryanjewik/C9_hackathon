import { useState, useEffect } from 'react';

interface PlayerStat {
  nickname: string;
  agents: string[];
  averageRating: number;
  kills: number;
  deaths: number;
  assists: number;
  firstKills: number;
  firstDeaths: number;
}

interface AgentStat {
  nickname: string;
  agent: string;
  averageRating: number;
  kills: number;
  deaths: number;
  assists: number;
  firstKills: number;
  firstDeaths: number;
}

type Mode = 'overall' | 'agent';

export function PlayerStats() {
  const [mode, setMode] = useState<Mode>('overall');
  const [players, setPlayers] = useState<PlayerStat[]>([]);
  const [agentStats, setAgentStats] = useState<AgentStat[]>([]);

  useEffect(() => {
    async function fetchPlayerStats() {
      try {
        const response = await fetch('/dashboard/player_stats');
        if (!response.ok) return;
        const data = await response.json();
        setPlayers(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error("Failed to load player stats: ", error);
      }
    }
    fetchPlayerStats();
  }, []);

  useEffect(() => {
    async function fetchAgentStats() {
      try {
        const response = await fetch('/dashboard/agent_stats');
        if (!response.ok) return;
        const data = await response.json();
        setAgentStats(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error("Failed to load agent stats: ", error);
      }
    }
    fetchAgentStats();
  }, []);

  return (
    <div className="mt-6 h-96 max-w-7xl rounded-2xl mx-auto bg-white bg-opacity-55 p-4 justify-items-center border-2 border-c9-cyan hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      <h1 className="text-2xl font-bold tracking-wide">
        <span className="text-c9-cyan font-extrabold">{"Players"}</span>
      </h1>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <h1>{mode === 'overall' ? 'Top 10 VCT Players (Last 10 Days)' : 'Best VCT Player Per Agent (Last 90 Days)'}</h1>
        <button
          onClick={() => setMode(mode === 'overall' ? 'agent' : 'overall')}
          style={{ fontSize: '12px', padding: '2px 8px', cursor: 'pointer' }}
        >
          {mode === 'overall' ? 'By Agent' : 'Overall'}
        </button>
      </div>
      <ul>
        {mode === 'overall'
          ? players.map((p) => (
              <li key={p.nickname}>
                {p.nickname} ({p.agents.join(', ')}) - Rating: {p.averageRating?.toFixed(2)} | K/D/A: {p.kills}/{p.deaths}/{p.assists} | FK/FD: {p.firstKills}/{p.firstDeaths}
              </li>
            ))
          : agentStats.map((a) => (
              <li key={a.agent}>
                [{a.agent}] {a.nickname} - Rating: {a.averageRating?.toFixed(2)} | K/D/A: {a.kills}/{a.deaths}/{a.assists} | FK/FD: {a.firstKills}/{a.firstDeaths}
              </li>
            ))
        }
      </ul>
    </div>
  );
}
