import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, Lock, Globe, Zap, Key } from 'lucide-react';

// ─── tiny helpers ────────────────────────────────────────────────────────────

function Method({ verb }: { verb: 'GET' | 'POST' | 'DELETE' | 'PATCH' }) {
  const map: Record<string, string> = {
    GET:    'bg-emerald-100 text-emerald-700',
    POST:   'bg-c9-cyan/15  text-c9-cyan',
    DELETE: 'bg-rose-100    text-rose-600',
    PATCH:  'bg-amber-100   text-amber-600',
  };
  return (
    <span className={`inline-block w-16 text-center text-xs font-extrabold rounded px-1.5 py-0.5 ${map[verb]}`}>
      {verb}
    </span>
  );
}

function Code({ children }: { children: string }) {
  return (
    <pre className="bg-c9-text/5 border border-c9-cyan/20 rounded-xl p-4 text-sm font-mono text-c9-text overflow-x-auto leading-relaxed whitespace-pre-wrap">
      {children}
    </pre>
  );
}

function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-6 ${className}`}>
      {children}
    </div>
  );
}

function SectionHeader({ children }: { children: ReactNode }) {
  return <h2 className="text-xl font-bold text-c9-text mb-4 flex items-center gap-2">{children}</h2>;
}

// ─── Collapsible endpoint row ─────────────────────────────────────────────────

interface Endpoint {
  method: 'GET' | 'POST' | 'DELETE' | 'PATCH';
  path: string;
  description: string;
  auth: boolean;
  params?: { name: string; in: 'query' | 'path' | 'body'; type: string; required: boolean; description: string }[];
  responseExample?: string;
  notes?: string;
}

function EndpointRow({ ep }: { ep: Endpoint }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-c9-cyan/20 rounded-2xl overflow-hidden">
      <button
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-c9-cyan/5 transition text-left"
        onClick={() => setOpen((v: boolean) => !v)}
      >
        <Method verb={ep.method} />
        <span className="font-mono text-sm text-c9-text flex-1">{ep.path}</span>
        {ep.auth
          ? <Lock className="w-3.5 h-3.5 text-c9-muted shrink-0" />
          : <Globe className="w-3.5 h-3.5 text-emerald-500 shrink-0" />}
        <span className="text-c9-muted text-sm hidden md:block flex-1 text-right truncate pr-2">{ep.description}</span>
        {open ? <ChevronDown className="w-4 h-4 text-c9-muted shrink-0" /> : <ChevronRight className="w-4 h-4 text-c9-muted shrink-0" />}
      </button>

      {open && (
        <div className="border-t border-c9-cyan/20 px-4 py-4 space-y-4 bg-white/60">
          <p className="text-c9-muted text-sm">{ep.description}</p>

          {ep.params && ep.params.length > 0 && (
            <div>
              <p className="text-xs font-bold text-c9-text uppercase tracking-widest mb-2">Parameters</p>
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="text-left text-c9-muted text-xs">
                    <th className="pb-1 pr-4 font-semibold">Name</th>
                    <th className="pb-1 pr-4 font-semibold">In</th>
                    <th className="pb-1 pr-4 font-semibold">Type</th>
                    <th className="pb-1 pr-4 font-semibold">Required</th>
                    <th className="pb-1 font-semibold">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {ep.params.map((p) => (
                    <tr key={p.name} className="border-t border-c9-cyan/10">
                      <td className="py-1.5 pr-4 font-mono text-c9-cyan">{p.name}</td>
                      <td className="py-1.5 pr-4 text-c9-muted">{p.in}</td>
                      <td className="py-1.5 pr-4 text-c9-muted">{p.type}</td>
                      <td className="py-1.5 pr-4">{p.required ? <span className="text-rose-500 font-bold">yes</span> : <span className="text-c9-muted">no</span>}</td>
                      <td className="py-1.5 text-c9-muted">{p.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {ep.responseExample && (
            <div>
              <p className="text-xs font-bold text-c9-text uppercase tracking-widest mb-2">Response Example</p>
              <Code>{ep.responseExample}</Code>
            </div>
          )}

          {ep.notes && (
            <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-xl px-3 py-2">{ep.notes}</p>
          )}
        </div>
      )}
    </div>
  );
}

function EndpointGroup({ title, icon, endpoints }: { title: string; icon?: ReactNode; endpoints: Endpoint[] }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="space-y-2">
      <button
        className="flex items-center gap-2 w-full text-left"
        onClick={() => setOpen((v: boolean) => !v)}
      >
        {icon}
        <h3 className="text-base font-bold text-c9-text">{title}</h3>
        {open ? <ChevronDown className="w-4 h-4 text-c9-muted ml-auto" /> : <ChevronRight className="w-4 h-4 text-c9-muted ml-auto" />}
      </button>
      {open && (
        <div className="space-y-1.5 pl-1">
          {endpoints.map((ep) => <EndpointRow key={ep.method + ep.path} ep={ep} />)}
        </div>
      )}
    </div>
  );
}

// ─── Data ─────────────────────────────────────────────────────────────────────

const paginationParams = [
  { name: 'page', in: 'query' as const, type: 'integer', required: false, description: 'Zero-based page index (default 0)' },
  { name: 'size', in: 'query' as const, type: 'integer', required: false, description: 'Page size (default 20)' },
];

// All paginated responses wrap content in a Spring Page envelope:
// { content: [...], totalElements: N, totalPages: N, number: 0, size: 20, first: true, last: false }

const endpointGroups: { title: string; icon: ReactNode; endpoints: Endpoint[] }[] = [
  {
    title: 'Tournaments',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/tournaments', auth: true,
        description: 'Paginated list of all tournaments.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 1,
      "name": "VCT Americas 2026",
      "tier": "S",
      "startDate": "2026-01-15",
      "endDate": "2026-04-30",
      "location": "Los Angeles, CA",
      "prizePool": "$250,000",
      "status": "completed"
    }
  ],
  "totalElements": 42,
  "totalPages": 3,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/tournaments/{id}', auth: true,
        description: 'Get a single tournament by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Tournament ID' }],
        responseExample: `{
  "id": 1,
  "name": "VCT Americas 2026",
  "tier": "S",
  "startDate": "2026-01-15",
  "endDate": "2026-04-30",
  "location": "Los Angeles, CA",
  "prizePool": "$250,000",
  "status": "completed"
}`,
      },
    ],
  },
  {
    title: 'Matches',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/matches', auth: true,
        description: 'Paginated list of all matches.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 45,
      "phase": "Playoffs",
      "date": "2026-04-20T18:00:00Z",
      "patch": "9.04",
      "tournamentId": 1,
      "tournamentName": "VCT Americas 2026",
      "team1Name": "Cloud9",
      "team1Id": 3,
      "team1Score": 2,
      "team2Name": "NRG Esports",
      "team2Id": 7,
      "team2Score": 0,
      "winner": 3,
      "format": "BO3",
      "map1": "Abyss",
      "map2": "Bind",
      "map3": null,
      "map4": null,
      "map5": null
    }
  ],
  "totalElements": 310,
  "totalPages": 16
}`,
      },
      {
        method: 'GET', path: '/api/matches/{id}', auth: true,
        description: 'Get a single match by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Match ID' }],
        responseExample: `{
  "id": 45,
  "phase": "Playoffs",
  "date": "2026-04-20T18:00:00Z",
  "patch": "9.04",
  "tournamentId": 1,
  "tournamentName": "VCT Americas 2026",
  "team1Name": "Cloud9",
  "team1Id": 3,
  "team1Score": 2,
  "team2Name": "NRG Esports",
  "team2Id": 7,
  "team2Score": 0,
  "winner": 3,
  "format": "BO3",
  "map1": "Abyss",
  "map2": "Bind",
  "map3": null,
  "map4": null,
  "map5": null
}`,
      },
    ],
  },
  {
    title: 'Players',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/players', auth: true,
        description: 'Paginated list of all players with all-time and last-60-day stats.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 7,
      "nickname": "mwzera",
      "firstName": "Leonardo",
      "lastName": "Gonçalves",
      "country": "Brazil",
      "teamId": 5,
      "titles": [1, 3],
      "all_time_maps": 210,
      "all_time_map_wins": 130,
      "all_time_map_losses": 80,
      "all_time_rating": 1.24,
      "all_time_acs": 248,
      "all_time_kills": 5320,
      "all_time_deaths": 4010,
      "all_time_assists": 980,
      "all_time_avg_kills": 25.3,
      "all_time_avg_deaths": 19.1,
      "all_time_avg_assists": 4.7,
      "all_time_kast": 73.4,
      "all_time_adr": 158.2,
      "all_time_hs_percent": 24.1,
      "all_time_fk": 420,
      "all_time_fd": 310,
      "all_time_avg_fk": 2.0,
      "all_time_avg_fd": 1.5,
      "last_60_maps": 18,
      "last_60_map_wins": 11,
      "last_60_map_losses": 7,
      "last_60_rating": 1.31,
      "last_60_acs": 262,
      "last_60_kills": 420,
      "last_60_deaths": 310,
      "last_60_assists": 85,
      "last_60_avg_kills": 23.3,
      "last_60_avg_deaths": 17.2,
      "last_60_avg_assists": 4.7,
      "last_60_kast": 76.2,
      "last_60_adr": 162.5,
      "last_60_hs_percent": 26.4,
      "last_60_fk": 32,
      "last_60_fd": 22,
      "last_60_avg_fk": 1.8,
      "last_60_avg_fd": 1.2
    }
  ],
  "totalElements": 150
}`,
      },
      {
        method: 'GET', path: '/api/players/{id}', auth: true,
        description: 'Get a single player by ID. Same shape as the list item above.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Player ID' }],
      },
    ],
  },
  {
    title: 'Teams',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/teams', auth: true,
        description: 'Paginated list of all esports teams.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 3,
      "name": "Cloud9",
      "teamTag": "C9",
      "location": "United States",
      "titles": [2],
      "matchWins": 45,
      "matchLosses": 28,
      "currentRosterId": 12
    }
  ],
  "totalElements": 30
}`,
      },
      {
        method: 'GET', path: '/api/teams/{id}', auth: true,
        description: 'Get a single team by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Team ID' }],
        responseExample: `{
  "id": 3,
  "name": "Cloud9",
  "teamTag": "C9",
  "location": "United States",
  "titles": [2],
  "matchWins": 45,
  "matchLosses": 28,
  "currentRosterId": 12
}`,
      },
    ],
  },
  {
    title: 'Rosters',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/rosters', auth: true,
        description: 'Paginated list of team rosters. Includes nested team and player objects.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 12,
      "teamId": 3,
      "player1": 14,
      "player2": 22,
      "player3": 7,
      "player4": 19,
      "player5": 31,
      "dateCreated": "2025-11-01",
      "mapWins": 58,
      "mapLosses": 34,
      "teamEntity": { "id": 3, "name": "Cloud9", ... },
      "player1Entity": { "id": 14, "nickname": "Zellsis", ... },
      "player2Entity": { ... },
      "player3Entity": { ... },
      "player4Entity": { ... },
      "player5Entity": { ... }
    }
  ]
}`,
      },
      {
        method: 'GET', path: '/api/rosters/{id}', auth: true,
        description: 'Get a roster by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Roster ID' }],
      },
    ],
  },
  {
    title: 'Player Games',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/player-games', auth: true,
        description: 'Paginated list of individual player performance records per map played.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 1001,
      "matchId": 45,
      "gameId": 201,
      "playerId": 7,
      "teamId": 3,
      "rosterId": 12,
      "tournamentId": 1,
      "map": "Abyss",
      "agent": "Jett",
      "rating": 1.41,
      "acs": 310,
      "kills": 28,
      "deaths": 17,
      "assists": 5,
      "kast": "78%",
      "adr": 182,
      "hsPercent": "31%",
      "fk": 4,
      "fd": 2
    }
  ]
}`,
      },
      {
        method: 'GET', path: '/api/player-games/{id}', auth: true,
        description: 'Get a single player game record.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Player game ID' }],
      },
    ],
  },
  {
    title: 'Game Scores',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/game-scores', auth: true,
        description: 'Paginated list of per-map game scores for each match.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 201,
      "matchId": 45,
      "team1Score": 13,
      "team2Score": 9,
      "team1Id": 3,
      "team2Id": 7,
      "team1Name": "Cloud9",
      "team2Name": "NRG Esports",
      "map": "Abyss",
      "winner": 3
    }
  ]
}`,
      },
      {
        method: 'GET', path: '/api/game-scores/{id}', auth: true,
        description: 'Get a single game score record.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Game score ID' }],
      },
    ],
  },
  {
    title: 'Map Veto',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/map-veto', auth: true,
        description: 'Paginated list of map veto actions per match. Includes nested team and match objects.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 301,
      "matchId": 45,
      "type": "ban",
      "teamId": 3,
      "mapSelected": "Haven",
      "turn": 1,
      "teamEntity": { "id": 3, "name": "Cloud9", ... },
      "matchEntity": { "id": 45, "phase": "Playoffs", ... }
    }
  ]
}`,
        notes: 'type is one of: "ban", "pick", "decider"',
      },
      {
        method: 'GET', path: '/api/map-veto/{id}', auth: true,
        description: 'Get a single map veto record.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Map veto ID' }],
      },
    ],
  },
  {
    title: 'Tournament Placements',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/tournament-placements', auth: true,
        description: 'Paginated list of team placements per tournament. Includes nested team and tournament objects.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 401,
      "tournamentId": 1,
      "placement": "1st",
      "esportsTeamId": 3,
      "prizeMoney": "$100,000",
      "stage": "Grand Final",
      "players": [14, 22, 7, 19, 31],
      "teamEntity": { "id": 3, "name": "Cloud9", ... },
      "tournamentEntity": { "id": 1, "name": "VCT Americas 2026", ... }
    }
  ]
}`,
      },
      {
        method: 'GET', path: '/api/tournament-placements/{id}', auth: true,
        description: 'Get a single tournament placement.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Placement ID' }],
      },
    ],
  },
  {
    title: 'Public Dashboard',
    icon: <Globe className="w-4 h-4 text-emerald-500" />,
    endpoints: [
      {
        method: 'GET', path: '/dashboard/ongoing_tournaments', auth: false,
        description: 'List all currently active tournaments.',
        responseExample: `[
  {
    "id": 1,
    "name": "VCT Americas 2026",
    "tier": "S",
    "startDate": "2026-01-15",
    "endDate": "2026-04-30",
    "location": "Los Angeles, CA",
    "status": "ongoing"
  }
]`,
      },
      {
        method: 'GET', path: '/dashboard/recent_matches', auth: false,
        description: 'List recent match results.',
      },
      {
        method: 'GET', path: '/dashboard/map_stats', auth: false,
        description: 'Map win/loss statistics for a specific team.',
        params: [{ name: 'teamId', in: 'query', type: 'integer', required: true, description: 'Team to query stats for' }],
      },
      {
        method: 'GET', path: '/dashboard/team_match_history', auth: false,
        description: 'Recent match history for a specific team.',
        params: [{ name: 'teamId', in: 'query', type: 'integer', required: true, description: 'Team to query history for' }],
      },
      {
        method: 'GET', path: '/dashboard/player_stats', auth: false,
        description: 'Top or bottom player performance stats.',
        params: [{ name: 'sort', in: 'query', type: '"top" | "bottom"', required: false, description: 'Sort direction — defaults to "top"' }],
      },
      {
        method: 'GET', path: '/dashboard/agent_stats', auth: false,
        description: 'Best performing player per agent.',
      },
      {
        method: 'GET', path: '/dashboard/agent_pickrates', auth: false,
        description: 'Pick rate statistics per agent across all matches.',
      },
      {
        method: 'GET', path: '/dashboard/tournament_map_stats', auth: false,
        description: 'Map pick and ban statistics aggregated across tournaments.',
      },
    ],
  },
];

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ApiDocs() {
  return (
    <div className="p-4 px-6 md:px-48 max-w-7xl mx-auto space-y-6 pb-16">

      {/* Hero */}
      <Card className="text-center">
        <h1 className="text-4xl font-extrabold text-c9-cyan mb-2 tracking-wide">API Reference</h1>
        <p className="text-c9-muted text-lg">Cloud9 Valorant Esports Data Platform</p>
      </Card>

      {/* Auth legend */}
      <div className="flex flex-wrap gap-4 text-sm text-c9-muted">
        <span className="flex items-center gap-1.5"><Lock className="w-3.5 h-3.5" /> Requires <code className="text-c9-text font-mono">Authorization: Bearer &lt;token&gt;</code></span>
        <span className="flex items-center gap-1.5"><Globe className="w-3.5 h-3.5 text-emerald-500" /> Public — no auth required</span>
      </div>

      {/* ── Authentication ───────────────────────────────────── */}
      <Card>
        <SectionHeader><Key className="w-5 h-5 text-c9-cyan" />Authentication</SectionHeader>
        <p className="text-c9-muted text-sm mb-4">
          All <code className="font-mono">/api/**</code> endpoints require a short-lived JWT obtained by exchanging your API key.
          API keys are managed through the dashboard.
        </p>
        <div className="space-y-4">
          <div>
            <p className="text-xs font-bold text-c9-text uppercase tracking-widest mb-2">1 — Exchange your API key for a token</p>
            <Code>{`POST /auth/token
Content-Type: application/json

{ "key": "your_api_key_here" }

// Response
{
  "access_token": "eyJhbGci...",
  "token_type": "Bearer",
  "expires_in": 900
}`}</Code>
          </div>
          <div>
            <p className="text-xs font-bold text-c9-text uppercase tracking-widest mb-2">2 — Attach the token to every request</p>
            <Code>{`GET /api/players
Authorization: Bearer eyJhbGci...`}</Code>
          </div>
          <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-xl px-3 py-2">
            Tokens expire after <strong>15 minutes</strong>. Re-call <code className="font-mono">POST /auth/token</code> to get a fresh one.
          </p>
        </div>
      </Card>

      {/* ── Rate Limiting ────────────────────────────────────── */}
      <Card>
        <SectionHeader><Zap className="w-5 h-5 text-amber-500" />Rate Limiting</SectionHeader>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
          {[
            { label: 'Window', value: '60 seconds' },
            { label: 'Max Requests', value: '100 / window' },
            { label: 'Exempt Routes', value: '/dashboard/**' },
          ].map((item) => (
            <div key={item.label} className="bg-c9-bg rounded-2xl border border-c9-cyan/20 p-4 text-center">
              <div className="text-2xl font-extrabold text-c9-cyan">{item.value}</div>
              <div className="text-xs text-c9-muted mt-1 uppercase tracking-widest">{item.label}</div>
            </div>
          ))}
        </div>
        <div className="space-y-2 text-sm">
          <p className="font-semibold text-c9-text">Response Headers</p>
          <table className="w-full text-c9-muted">
            <tbody>
              {[
                ['X-RateLimit-Limit',     'Total requests allowed in the current window'],
                ['X-RateLimit-Remaining', 'Requests remaining before throttling'],
                ['Retry-After',           'Seconds until the window resets (only on 429)'],
              ].map(([h, d]) => (
                <tr key={h} className="border-t border-c9-cyan/10">
                  <td className="py-1.5 pr-4 font-mono text-c9-cyan">{h}</td>
                  <td className="py-1.5">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 rounded-xl bg-rose-50 border border-rose-200 px-4 py-2 text-sm text-rose-600">
          When the limit is exceeded the server returns <strong>429 Too Many Requests</strong>. Back off for the duration in <code className="font-mono">Retry-After</code>.
        </div>
      </Card>

      {/* ── Endpoint Reference ───────────────────────────────── */}
      <Card>
        <SectionHeader>Endpoint Reference</SectionHeader>
        <div className="space-y-6">
          {endpointGroups.map((g) => (
            <EndpointGroup key={g.title} title={g.title} icon={g.icon} endpoints={g.endpoints} />
          ))}
        </div>
      </Card>

    </div>
  );
}
