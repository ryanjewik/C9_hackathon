import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ComposedChart, Line, Cell,
} from 'recharts';

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
  format: string;
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

interface MatchHistoryEntry {
  date: string;
  won: boolean;
  opponentName: string;
  teamScore: number;
  opponentScore: number;
  tournamentName: string;
}

const PICK_COLOR = '#4dd9e8';
const BAN_COLOR  = '#e74c3c';
const WIN_COLOR  = '#1abc9c';
const LOSS_COLOR = '#e74c3c';
const TREND_COLOR = '#f1c40f';

function teamIconSrc(name: string): string {
  const normalized = name.toLowerCase().replace(/[\s.]/g, '_').replace(/_+/g, '_');
  return `/teams/${normalized}.png`;
}

function TeamScore({ name, score, isWinner }: { name: string; score: number; isWinner: boolean }) {
  return (
    <div
      className="flex flex-col items-center gap-1 px-6 py-3 rounded-xl ring-2 transition"
      style={{
        backgroundColor: isWinner ? 'rgba(26,188,156,0.15)' : 'rgba(231,76,60,0.12)',
        outline: `2px solid ${isWinner ? '#1abc9c' : '#e74c3c'}`,
      }}
    >
      <img
        src={teamIconSrc(name)}
        alt={name}
        onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
        className="w-12 h-12 object-contain mb-1"
      />
      <span className="text-base font-bold text-c9-text text-center leading-tight">{name}</span>
      <span
        className="text-4xl font-extrabold"
        style={{ color: isWinner ? '#1abc9c' : '#e74c3c' }}
      >
        {score}
      </span>
      <span
        className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
        style={{
          backgroundColor: isWinner ? '#1abc9c' : '#e74c3c',
          color: '#fff',
        }}
      >
        {isWinner ? '✓ Winner' : '✗ Loser'}
      </span>
    </div>
  );
}

interface TooltipProps {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: string;
}
function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-c9-cyan rounded-lg px-3 py-2 shadow text-sm text-c9-text">
      <div className="font-semibold mb-1">{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>{p.name}: <span className="font-bold">{p.value}</span></div>
      ))}
    </div>
  );
}

interface TrendTooltipProps {
  active?: boolean;
  payload?: { name: string; value: number; color: string; payload: { won: boolean; score: string; opponent: string } }[];
  label?: number;
}
function TrendTooltip({ active, payload }: TrendTooltipProps) {
  if (!active || !payload?.length) return null;
  const d = (payload.find(p => p.name === 'W/L') ?? payload[0])?.payload;
  return (
    <div className="bg-white border border-c9-cyan rounded-lg px-3 py-2 shadow text-sm text-c9-text">
      <div className="font-semibold mb-1">vs {d?.opponent}</div>
      <div style={{ color: d?.won ? WIN_COLOR : LOSS_COLOR }} className="font-bold">{d?.won ? 'Win' : 'Loss'} — {d?.score}</div>
      {payload.find(p => p.name === '5-game Win %') && (
        <div style={{ color: TREND_COLOR }}>
          5-game Win %: <span className="font-bold">{payload.find(p => p.name === '5-game Win %')!.value}%</span>
        </div>
      )}
    </div>
  );
}

function FormDots({ history }: { history: MatchHistoryEntry[] }) {
  // Show last 10, oldest→newest left→right
  const recent = [...history].reverse().slice(-10);
  return (
    <div className="flex items-center gap-1.5 justify-center flex-wrap">
      {recent.map((m, i) => (
        <div
          key={i}
          title={`vs ${m.opponentName} — ${m.won ? 'W' : 'L'} ${m.teamScore}-${m.opponentScore}`}
          className="w-5 h-5 rounded-full flex items-center justify-center text-white font-bold text-[9px] cursor-default"
          style={{ backgroundColor: m.won ? WIN_COLOR : LOSS_COLOR }}
        >
          {m.won ? 'W' : 'L'}
        </div>
      ))}
    </div>
  );
}

function WinRateTrendChart({ teamName, history }: { teamName: string; history: MatchHistoryEntry[] }) {
  // Oldest→newest, rolling 5-match win %
  const ordered = [...history].reverse();
  const data = ordered.map((m, i) => {
    const window = ordered.slice(Math.max(0, i - 4), i + 1);
    const winRate = Math.round((window.filter(w => w.won).length / window.length) * 100);
    const short = m.opponentName.length > 8 ? m.opponentName.slice(0, 7) + '…' : m.opponentName;
    return {      idx: i,      opponent: short,
      result: 1,
      winRate,
      won: m.won,
      score: `${m.teamScore}-${m.opponentScore}`,
    };
  });

  if (data.length === 0) return <div className="flex-1 flex items-center justify-center text-xs text-c9-muted">No history</div>;

  return (
    <div className="flex-1 min-w-0">
      <p className="text-xs font-semibold text-c9-muted uppercase tracking-wide text-center mb-1">{teamName} — Form & Win Rate</p>
      {/* Form dots */}
      <div className="mb-2">
        <FormDots history={history} />
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 24, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d0e8f0" />
          <XAxis dataKey="idx" tickFormatter={(i: number) => data[i]?.opponent ?? ''} tick={{ fontSize: 9, fill: '#6b8ca8' }} tickLine={false} angle={-30} textAnchor="end" interval={0} />
          <YAxis yAxisId="left" domain={[0, 1]} tickFormatter={() => ''} tickLine={false} axisLine={false} width={16} />
          <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 9, fill: '#6b8ca8' }} tickLine={false} axisLine={false} width={32} />
          <Tooltip content={<TrendTooltip />} cursor={{ fill: 'rgba(77,217,232,0.08)' }} />
          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
          <Bar yAxisId="left" dataKey="result" name="W/L" radius={[4, 4, 0, 0]} fill={WIN_COLOR} label={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.won ? WIN_COLOR : LOSS_COLOR} />
            ))}
          </Bar>
          <Line yAxisId="right" type="monotone" dataKey="winRate" name="5-game Win %" stroke={TREND_COLOR} strokeWidth={2} dot={{ r: 3, fill: TREND_COLOR }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function MapStatsChart({ teamName, stats }: { teamName: string; stats: MapStat[] }) {
  const data = [...stats].sort((a, b) => (b.pickCount + b.banCount) - (a.pickCount + a.banCount));
  return (
    <div className="flex-1 min-w-0">
      <p className="text-xs font-semibold text-c9-muted uppercase tracking-wide text-center mb-1">{teamName} — Map Tendencies</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 24, left: 0 }} barCategoryGap="30%">
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d0e8f0" />
          <XAxis dataKey="mapSelected" tick={{ fontSize: 10, fill: '#6b8ca8' }} tickLine={false} angle={-35} textAnchor="end" interval={0} />
          <YAxis tick={{ fontSize: 10, fill: '#6b8ca8' }} tickLine={false} axisLine={false} allowDecimals={false} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(77,217,232,0.08)' }} />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
          <Bar dataKey="pickCount" name="Picks" fill={PICK_COLOR} radius={[4, 4, 0, 0]} />
          <Bar dataKey="banCount"  name="Bans"  fill={BAN_COLOR}  radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RecentMatches() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [mapStats, setMapStats] = useState<Record<number, MapStat[]>>({});
  const [matchHistory, setMatchHistory] = useState<Record<number, MatchHistoryEntry[]>>({});
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    async function fetchAll() {
      try {
        const response = await fetch('/dashboard/recent_matches');
        if (!response.ok) return;
        const data: Match[] = await response.json();
        setMatches(Array.isArray(data) ? data : []);

        const teamIds = [...new Set(data.flatMap((m) => [m.team1Id, m.team2Id]))];

        const [statsEntries, historyEntries] = await Promise.all([
          Promise.all(
            teamIds.map(async (teamId) => {
              const res = await fetch(`/dashboard/map_stats?teamId=${teamId}`);
              if (!res.ok) return [teamId, []] as [number, MapStat[]];
              return [teamId, await res.json()] as [number, MapStat[]];
            }),
          ),
          Promise.all(
            teamIds.map(async (teamId) => {
              const res = await fetch(`/dashboard/team_match_history?teamId=${teamId}`);
              if (!res.ok) return [teamId, []] as [number, MatchHistoryEntry[]];
              return [teamId, await res.json()] as [number, MatchHistoryEntry[]];
            }),
          ),
        ]);

        setMapStats(Object.fromEntries(statsEntries));
        setMatchHistory(Object.fromEntries(historyEntries));
      } catch (error) {
        console.error('Failed to load recent matches:', error);
      }
    }
    fetchAll();
  }, []);

  const match = matches[activeIdx];
  const playedMaps = useMemo(() => {
    if (!match) return [];
    return [match.map1, match.map2, match.map3, match.map4, match.map5].filter(Boolean) as string[];
  }, [match]);

  return (
    <div className="h-full rounded-2xl bg-white bg-opacity-55 p-4 border-2 border-c9-cyan hover:shadow-lg hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      {/* Header */}
      <div className="flex flex-col gap-2 mb-3">
        <h2 className="text-2xl font-bold tracking-wide">
          <span className="text-c9-cyan font-extrabold">Recent Matches</span>
        </h2>
        <div className="flex gap-2 flex-wrap justify-center">
          {matches.map((m, i) => (
            <button
              key={m.id}
              onClick={() => setActiveIdx(i)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-semibold border transition ${
                i === activeIdx
                  ? 'bg-c9-cyan text-white border-c9-cyan'
                  : 'border-c9-cyan text-c9-cyan bg-transparent hover:bg-c9-cyan hover:bg-opacity-10'
              }`}
            >
              <img src={teamIconSrc(m.team1Name)} alt="" onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }} className="w-4 h-4 object-contain" />
              {m.team1Name} vs {m.team2Name}
              <img src={teamIconSrc(m.team2Name)} alt="" onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }} className="w-4 h-4 object-contain" />
            </button>
          ))}
        </div>
      </div>

      {match && (
        <>
          {/* Tournament / phase / date */}
          <div className="text-center text-xs text-c9-muted mb-3 space-x-2">
            <span className="font-semibold text-c9-text">{match.tournamentName}</span>
            <span>·</span>
            <span>{match.phase}</span>
            <span>·</span>
            <span>{match.format.toUpperCase()}</span>
            <span>·</span>
            <span>{new Date(match.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</span>
          </div>

          {/* Score row */}
          <div className="flex items-center justify-center gap-4 mb-4">
            <TeamScore name={match.team1Name} score={match.team1Score} isWinner={match.winner === match.team1Id} />
            <span className="text-2xl font-black text-c9-muted">vs</span>
            <TeamScore name={match.team2Name} score={match.team2Score} isWinner={match.winner === match.team2Id} />
          </div>

          {/* Played maps */}
          {playedMaps.length > 0 && (
            <div className="flex justify-center gap-2 mb-5 flex-wrap">
              {playedMaps.map((map, i) => (
                <span
                  key={i}
                  className="px-3 py-1 rounded-full text-xs font-semibold border border-c9-cyan text-c9-text bg-white bg-opacity-60"
                >
                  {map}
                </span>
              ))}
            </div>
          )}

          {/* Win rate trend charts */}
          <div className="mb-2">
            <p className="text-xs font-semibold text-c9-muted uppercase tracking-widest text-center mb-2">Season Form &amp; Win Rate Trend</p>
            <div className="flex flex-col md:flex-row gap-4">
              <WinRateTrendChart teamName={match.team1Name} history={matchHistory[match.team1Id] ?? []} />
              <div className="hidden md:block w-px bg-c9-cyan opacity-20 self-stretch" />
              <WinRateTrendChart teamName={match.team2Name} history={matchHistory[match.team2Id] ?? []} />
            </div>
          </div>

          <div className="my-3 border-t border-c9-cyan opacity-20" />

          {/* Map tendency charts */}
          <p className="text-xs font-semibold text-c9-muted uppercase tracking-widest text-center mb-2">Map Tendencies</p>
          <div className="flex flex-col md:flex-row gap-4">
            <MapStatsChart teamName={match.team1Name} stats={mapStats[match.team1Id] ?? []} />
            <div className="hidden md:block w-px bg-c9-cyan opacity-20 self-stretch" />
            <MapStatsChart teamName={match.team2Name} stats={mapStats[match.team2Id] ?? []} />
          </div>
        </>
      )}
    </div>
  );
}
