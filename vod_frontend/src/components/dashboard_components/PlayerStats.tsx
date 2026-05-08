import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

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
type View = 'table' | 'chart';

const C9_CYAN = '#4dd9e8';
const C9_BLUE = '#5bbfe8';

export function PlayerStats() {
  const [mode, setMode] = useState<Mode>('overall');
  const [view, setView] = useState<View>('table');
  const [players, setPlayers] = useState<PlayerStat[]>([]);
  const [agentStats, setAgentStats] = useState<AgentStat[]>([]);

  useEffect(() => {
    async function fetchPlayerStats() {
      try {
        const response = await fetch('/dashboard/player_stats');
        if (!response.ok) return;
        const data = await response.json();
        console.log('[PlayerStats] player_stats response:', JSON.stringify(data?.slice?.(0,2)));
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

  type ChartEntry = { name: string; rating: number; agents: string; agent: string };
  const chartData: ChartEntry[] = mode === 'overall'
    ? players.map((p) => ({ name: p.nickname, rating: p.averageRating ?? 0, agents: (p.agents ?? []).join(', '), agent: '' }))
    : agentStats.map((a) => ({ name: a.nickname, rating: a.averageRating ?? 0, agent: a.agent, agents: '' }));

  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: {value: number; payload: {agents?: string; agent?: string}}[]; label?: string }) => {
    if (!active || !payload?.length) return null;
    const agentsList = payload[0].payload.agents ?? '';
    const singleAgent = payload[0].payload.agent ?? '';
    const agentNames = agentsList ? agentsList.split(', ').filter(Boolean) : singleAgent ? [singleAgent] : [];
    return (
      <div style={{ borderRadius: '8px', border: `1px solid ${C9_CYAN}`, background: 'rgba(255,255,255,0.95)', padding: '8px 12px', fontSize: 12 }}>
        <p style={{ fontWeight: 600, color: '#1a2a3a', marginBottom: 4 }}>{label}</p>
        {agentNames.length > 0 && (
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 4 }}>
            {agentNames.map((a) => <AgentIcon key={a} agent={a} size={20} />)}
            <span style={{ color: '#6b8ca8', fontSize: 11 }}>{agentNames.join(', ')}</span>
          </div>
        )}
        <p style={{ color: C9_CYAN }}>Avg Rating: {typeof payload[0].value === 'number' ? payload[0].value.toFixed(2) : payload[0].value}</p>
      </div>
    );
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const AgentAxisTick = ({ x, y, payload }: any) => {
    const entry = chartData.find((d) => d.name === payload.value);
    const agent = entry?.agent ?? '';
    return (
      <g transform={`translate(${x},${y})`}>
        {agent && (
          <image href={`/agents/${agent.toLowerCase().trim()}.webp`} x={-10} y={2} width={20} height={20} />
        )}
        <text x={0} y={agent ? 28 : 12} textAnchor="middle" fontSize={10} fill="#1a2a3a">
          {payload.value}
        </text>
      </g>
    );
  };

  return (
    <div className="mt-6 max-w-7xl rounded-2xl mx-auto bg-white bg-opacity-55 p-4 border-2 border-c9-cyan hover:shadow-lg hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h1 className="text-2xl font-bold tracking-wide">
          <span className="text-c9-cyan font-extrabold">Players</span>
        </h1>
        <div className="flex gap-2">
          {/* Table / Chart toggle */}
          <div className="flex rounded-lg overflow-hidden border border-c9-cyan text-xs font-semibold">
            <button
              onClick={() => setView('table')}
              className={`px-3 py-1 transition-colors ${view === 'table' ? 'bg-c9-cyan text-white' : 'bg-transparent text-c9-cyan hover:bg-c9-cyan hover:bg-opacity-20'}`}
            >
              Table
            </button>
            <button
              onClick={() => setView('chart')}
              className={`px-3 py-1 transition-colors ${view === 'chart' ? 'bg-c9-cyan text-white' : 'bg-transparent text-c9-cyan hover:bg-c9-cyan hover:bg-opacity-20'}`}
            >
              Chart
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
        {mode === 'overall' ? 'Top 10 VCT Players — Last 10 Days' : 'Best VCT Player Per Agent — Last 90 Days'}
      </p>

      {/* Table view */}
      {view === 'table' && (
        <div className="overflow-auto max-h-72">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-b-2 border-c9-cyan text-c9-cyan text-xs uppercase tracking-wider">
                <th className="py-2 pr-4 font-semibold">Player</th>
                {mode === 'overall' && <th className="py-2 pr-4 font-semibold">Agents</th>}
                {mode === 'agent'  && <th className="py-2 pr-4 font-semibold">Agent</th>}
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
                      <td className="py-1.5 pr-4">
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
      )}

      {/* Chart view */}
      {view === 'chart' && (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#4dd9e820" />
              <XAxis
                dataKey="name"
                tick={mode === 'agent' ? <AgentAxisTick /> : { fontSize: 11, fill: '#1a2a3a' }}
                angle={mode === 'agent' ? 0 : -35}
                textAnchor={mode === 'agent' ? 'middle' : 'end'}
                interval={0}
                height={mode === 'agent' ? 50 : 40}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#6b8ca8' }}
                domain={['auto', 'auto']}
                label={{ value: 'Rating', angle: -90, position: 'insideLeft', offset: 14, fontSize: 11, fill: '#6b8ca8' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="rating" radius={[4, 4, 0, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={i % 2 === 0 ? C9_CYAN : C9_BLUE} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
