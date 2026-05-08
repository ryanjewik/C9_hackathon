import { useState, useEffect } from 'react';

function AgentIcon({ agent, size = 22 }: { agent: string; size?: number }) {
  const src = `/agents/${agent.toLowerCase().trim()}.webp`;
  return (
    <img
      src={src}
      alt={agent}
      title={agent}
      width={size}
      height={size}
      className="inline-block rounded-full object-cover"
      onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
    />
  );
}

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
type Sort = 'top' | 'bottom';

export function PlayerStats() {
  const [mode, setMode] = useState<Mode>('overall');
  const [sort, setSort] = useState<Sort>('top');
  const [players, setPlayers] = useState<PlayerStat[]>([]);
  const [agentStats, setAgentStats] = useState<AgentStat[]>([]);

  useEffect(() => {
    async function fetchPlayerStats() {
      try {
        const response = await fetch(`/dashboard/player_stats?sort=${sort}`);
        if (!response.ok) return;
        const data = await response.json();
        console.log('[PlayerStats] player_stats response:', JSON.stringify(data?.slice?.(0,2)));
        setPlayers(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error("Failed to load player stats: ", error);
      }
    }
    fetchPlayerStats();
  }, [sort]);

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
    <div className="h-full rounded-2xl bg-white bg-opacity-55 p-4 border-2 border-c9-cyan hover:shadow-lg hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h1 className="text-2xl font-bold tracking-wide">
          <span className="text-c9-cyan font-extrabold">Players</span>
        </h1>
        <div className="flex gap-2">
          {/* Top / Bottom toggle */}
          <div className="flex rounded-lg overflow-hidden border border-c9-cyan text-xs font-semibold">
            <button
              onClick={() => setSort('top')}
              className={`px-3 py-1 transition-colors ${sort === 'top' ? 'bg-c9-cyan text-white' : 'bg-transparent text-c9-cyan hover:bg-c9-cyan hover:bg-opacity-20'}`}
            >
              Top 25
            </button>
            <button
              onClick={() => setSort('bottom')}
              className={`px-3 py-1 transition-colors ${sort === 'bottom' ? 'bg-c9-cyan text-white' : 'bg-transparent text-c9-cyan hover:bg-c9-cyan hover:bg-opacity-20'}`}
            >
              Bottom 25
            </button>
          </div>
          {/* Overall / By Agent toggle */}
          <div className="flex rounded-lg overflow-hidden border border-c9-cyan text-xs font-semibold">
            <button
              onClick={() => setMode('overall')}
              className={`px-3 py-1 transition-colors ${mode === 'overall' ? 'bg-c9-cyan text-white' : 'bg-transparent text-c9-cyan hover:bg-c9-cyan hover:bg-opacity-20'}`}
            >
              Overall
            </button>
            <button
              onClick={() => setMode('agent')}
              className={`px-3 py-1 transition-colors ${mode === 'agent' ? 'bg-c9-cyan text-white' : 'bg-transparent text-c9-cyan hover:bg-c9-cyan hover:bg-opacity-20'}`}
            >
              By Agent
            </button>
          </div>
        </div>
      </div>
      <p className="text-xs text-c9-muted mb-3">
        {mode === 'overall'
          ? `${sort === 'top' ? 'Top' : 'Bottom'} 25 VCT Players — Last 2 Weeks`
          : 'Best VCT Player Per Agent — Last 90 Days'}
      </p>

      <div className="pb-4 pr-1">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-b-2 border-c9-cyan text-c9-cyan text-xs uppercase tracking-wider">
                <th className="py-2 pr-4 font-semibold">Player</th>
                {mode === 'overall' && <th className="py-2 pr-4 font-semibold w-36">Agents</th>}
                {mode === 'agent'  && <th className="py-2 pr-4 font-semibold w-28">Agent</th>}
                <th className="py-2 pr-4 font-semibold text-right">Rating</th>
                <th className="py-2 pr-4 font-semibold text-right">K</th>
                <th className="py-2 pr-4 font-semibold text-right">D</th>
                <th className="py-2 pr-4 font-semibold text-right">A</th>
                <th className="py-2 pr-4 font-semibold text-right">FK</th>
                <th className="py-2 font-semibold text-right">FD</th>
              </tr>
            </thead>
            <tbody>
              {mode === 'overall'
                ? players.map((p, i) => (
                    <tr
                      key={p.nickname}
                      className={`border-b border-c9-cyan border-opacity-20 ${i % 2 === 0 ? 'bg-transparent' : 'bg-c9-cyan bg-opacity-5'} hover:bg-c9-cyan hover:bg-opacity-10 transition-colors`}
                    >
                      <td className="py-1.5 pr-4 font-medium text-c9-text">{p.nickname}</td>
                      <td className="py-1.5 pr-4 w-36">
                        <div className="flex gap-1 flex-wrap items-center">
                          {(p.agents ?? []).map((a) => (
                            <span key={a} className="flex items-center gap-0.5 text-xs text-c9-muted">
                              <AgentIcon agent={a} size={18} />
                              {a}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="py-1.5 pr-4 text-right font-semibold text-c9-cyan">{p.averageRating?.toFixed(2)}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{p.kills}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{p.deaths}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{p.assists}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{p.firstKills}</td>
                      <td className="py-1.5 text-right text-c9-text">{p.firstDeaths}</td>
                    </tr>
                  ))
                : agentStats.map((a, i) => (
                    <tr
                      key={a.agent}
                      className={`border-b border-c9-cyan border-opacity-20 ${i % 2 === 0 ? 'bg-transparent' : 'bg-c9-cyan bg-opacity-5'} hover:bg-c9-cyan hover:bg-opacity-10 transition-colors`}
                    >
                      <td className="py-1.5 pr-4 font-medium text-c9-text">{a.nickname}</td>
                      <td className="py-1.5 pr-4">
                        <div className="flex items-center gap-1.5">
                          <AgentIcon agent={a.agent} size={22} />
                          <span className="text-c9-muted text-xs">{a.agent}</span>
                        </div>
                      </td>
                      <td className="py-1.5 pr-4 text-right font-semibold text-c9-cyan">{a.averageRating?.toFixed(2)}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{a.kills}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{a.deaths}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{a.assists}</td>
                      <td className="py-1.5 pr-4 text-right text-c9-text">{a.firstKills}</td>
                      <td className="py-1.5 text-right text-c9-text">{a.firstDeaths}</td>
                    </tr>
                  ))
              }
            </tbody>
          </table>
        </div>
    </div>
  );
}
