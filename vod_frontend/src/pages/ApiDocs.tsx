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

function EndpointGroup({ title, description, icon, endpoints }: { title: string; description?: string; icon?: ReactNode; endpoints: Endpoint[] }) {
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
      {description && <p className="text-sm text-c9-muted pl-1">{description}</p>}
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

const endpointGroups: { title: string; description?: string; icon: ReactNode; endpoints: Endpoint[] }[] = [
  {
    title: 'Tournaments',
    description: 'All recorded Valorant esports tournaments spanning VCT, Challengers, and Offseason Events.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/tournaments', auth: true,
        description: 'Paginated list of all tournaments.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 2283,
      "name": "Valorant Champions 2025",
      "tier": "VCT",
      "start_date": "2025-09-12",
      "end_date": "2025-10-05",
      "location": "Accor Arena, Paris",
      "prize_pool": "$2,250,000 USD",
      "status": "completed"
    }
  ],
  "totalElements": 749,
  "totalPages": 749,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/tournaments/{id}', auth: true,
        description: 'Get a single tournament by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Tournament ID' }],
        responseExample: `{
  "id": 2283,
  "name": "Valorant Champions 2025",
  "tier": "VCT",
  "start_date": "2025-09-12",
  "end_date": "2025-10-05",
  "location": "Accor Arena, Paris",
  "prize_pool": "$2,250,000 USD",
  "status": "completed"
}`,
      },
    ],
  },
  {
    title: 'Matches',
    description: 'Match records with scores, format, maps played, and links to per-map game scores and veto data.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/matches', auth: true,
        description: 'Paginated list of all matches.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 582531,
      "phase": "Main Event: Quarterfinals",
      "date": "2025-11-12T04:00:00Z",
      "patch": null,
      "tournamentId": 2720,
      "tournamentName": "China Evolution Series: Epilogue",
      "team1Name": "Nova Esports",
      "team1Id": 12064,
      "team1Score": 2,
      "team2Name": "EDward Gaming",
      "team2Id": 1120,
      "team2Score": 1,
      "winner": 12064,
      "format": "bo3",
      "map1": "Split",
      "map2": "Sunset",
      "map3": "Haven",
      "map4": null,
      "map5": null,
      "gameScoreIds": [240467, 240468, 240469],
      "mapVetoIds": [1, 2, 3, 4, 5, 6]
    }
  ],
  "totalElements": 32395,
  "totalPages": 32395,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/matches/{id}', auth: true,
        description: 'Get a single match by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Match ID' }],
        responseExample: `{
  "id": 582531,
  "phase": "Main Event: Quarterfinals",
  "date": "2025-11-12T04:00:00Z",
  "patch": null,
  "tournamentId": 2720,
  "tournamentName": "China Evolution Series: Epilogue",
  "team1Name": "Nova Esports",
  "team1Id": 12064,
  "team1Score": 2,
  "team2Name": "EDward Gaming",
  "team2Id": 1120,
  "team2Score": 1,
  "winner": 12064,
  "format": "bo3",
  "map1": "Split",
  "map2": "Sunset",
  "map3": "Haven",
  "map4": null,
  "map5": null,
  "gameScoreIds": [240467, 240468, 240469],
  "mapVetoIds": [1, 2, 3, 4, 5, 6]
}`,
      },
    ],
  },
  {
    title: 'Players',
    description: 'Player profiles with career-aggregate stats computed across all recorded maps.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/players', auth: true,
        description: 'Paginated list of all players with all-time stats.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 42,
      "nickname": "KDS",
      "country": "United States",
      "team_id": 1281,
      "titles": [],
      "all_time_maps": 7,
      "all_time_map_wins": 2,
      "all_time_map_losses": 5,
      "all_time_rating": 0.69,
      "all_time_acs": 175.29,
      "all_time_kills": 77,
      "all_time_deaths": 112,
      "all_time_assists": 24,
      "all_time_avg_kills": 11.00,
      "all_time_avg_deaths": 16.00,
      "all_time_avg_assists": 3.43,
      "all_time_kast": 61.57,
      "all_time_adr": 117.86,
      "all_time_hs_percent": 15.86,
      "all_time_fk": 23,
      "all_time_fd": 26,
      "all_time_avg_fk": 3.29,
      "all_time_avg_fd": 3.71
    }
  ],
  "totalElements": 29022,
  "totalPages": 29022,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/players/{id}', auth: true,
        description: 'Get a single player by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Player ID' }],
        responseExample: `{
  "id": 42,
  "nickname": "KDS",
  "country": "United States",
  "team_id": 1281,
  "titles": [],
  "all_time_maps": 7,
  "all_time_map_wins": 2,
  "all_time_map_losses": 5,
  "all_time_rating": 0.69,
  "all_time_acs": 175.29,
  "all_time_kills": 77,
  "all_time_deaths": 112,
  "all_time_assists": 24,
  "all_time_avg_kills": 11.00,
  "all_time_avg_deaths": 16.00,
  "all_time_avg_assists": 3.43,
  "all_time_kast": 61.57,
  "all_time_adr": 117.86,
  "all_time_hs_percent": 15.86,
  "all_time_fk": 23,
  "all_time_fd": 26,
  "all_time_avg_fk": 3.29,
  "all_time_avg_fd": 3.71
}`,
      },
    ],
  },
  {
    title: 'Teams',
    description: 'Esports organizations with overall match win/loss records and roster history.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/teams', auth: true,
        description: 'Paginated list of all esports teams.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 387,
      "name": "QQQ Hasagi",
      "teamTag": "QQQ",
      "location": "Vietnam",
      "titles": [],
      "matchWins": 0,
      "matchLosses": 0
    }
  ],
  "totalElements": 10586,
  "totalPages": 10586,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/teams/{id}', auth: true,
        description: 'Get a single team by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Team ID' }],
        responseExample: `{
  "id": 387,
  "name": "QQQ Hasagi",
  "teamTag": "QQQ",
  "location": "Vietnam",
  "titles": [],
  "matchWins": 0,
  "matchLosses": 0
}`,
      },
    ],
  },
  {
    title: 'Rosters',
    description: 'Five-player roster snapshots tied to a team at a specific point in time, with map win/loss records.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/rosters', auth: true,
        description: 'Paginated list of team rosters. Includes nested team and player objects.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 2,
      "dateCreated": "2025-11-12",
      "mapWins": 1,
      "mapLosses": 2,
      "team": { "id": 1120, "name": "EDward Gaming", "teamTag": "EDG" },
      "player1": { "id": 34180, "nickname": "P1n" },
      "player2": { "id": 56277, "nickname": "AnJing" },
      "player3": { "id": 59796, "nickname": "YoungX" },
      "player4": { "id": 59797, "nickname": "Moonlight" },
      "player5": { "id": 59798, "nickname": "ggd" }
    }
  ],
  "totalElements": 15508,
  "totalPages": 15508,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/rosters/{id}', auth: true,
        description: 'Get a roster by ID.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Roster ID' }],
        responseExample: `{
  "id": 2,
  "dateCreated": "2025-11-12",
  "mapWins": 1,
  "mapLosses": 2,
  "team": { "id": 1120, "name": "EDward Gaming", "teamTag": "EDG" },
  "player1": { "id": 34180, "nickname": "P1n" },
  "player2": { "id": 56277, "nickname": "AnJing" },
  "player3": { "id": 59796, "nickname": "YoungX" },
  "player4": { "id": 59797, "nickname": "Moonlight" },
  "player5": { "id": 59798, "nickname": "ggd" }
}`,
      },
    ],
  },
  {
    title: 'Player Games',
    description: 'Per-player, per-map performance records: ACS, K/D/A, ADR, headshot %, first kills/deaths, and more.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/player-games', auth: true,
        description: 'Paginated list of individual player performance records per map played.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 1,
      "matchId": 582531,
      "gameId": 240467,
      "teamId": 12064,
      "rosterId": 1,
      "tournamentId": 2720,
      "map": "Split",
      "agent": "skye",
      "rating": null,
      "acs": 290,
      "kills": 17,
      "deaths": 9,
      "assists": 11,
      "kast": null,
      "adr": null,
      "hsPercent": null,
      "fk": null,
      "fd": null,
      "player": { "id": 4712, "nickname": "heybay" }
    }
  ],
  "totalElements": 606545,
  "totalPages": 606545,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/player-games/{id}', auth: true,
        description: 'Get a single player game record.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Player game ID' }],
        responseExample: `{
  "id": 1,
  "matchId": 582531,
  "gameId": 240467,
  "teamId": 12064,
  "rosterId": 1,
  "tournamentId": 2720,
  "map": "Split",
  "agent": "skye",
  "rating": null,
  "acs": 290,
  "kills": 17,
  "deaths": 9,
  "assists": 11,
  "kast": null,
  "adr": null,
  "hsPercent": null,
  "fk": null,
  "fd": null,
  "player": { "id": 4712, "nickname": "heybay" }
}`,
      },
    ],
  },
  {
    title: 'Game Scores',
    description: 'Per-map round scores for each match, linking back to all player game records for that map.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/game-scores', auth: true,
        description: 'Paginated list of per-map game scores for each match.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 240467,
      "matchId": 582531,
      "team1Score": 13,
      "team2Score": 3,
      "team1Id": 12064,
      "team2Id": 1120,
      "team1Name": "Nova Esports",
      "team2Name": "EDward Gaming",
      "map": "Split",
      "winner": 12064,
      "playerGameIds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
  ],
  "totalElements": 61013,
  "totalPages": 61013,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/game-scores/{id}', auth: true,
        description: 'Get a single game score record.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Game score ID' }],
        responseExample: `{
  "id": 240467,
  "matchId": 582531,
  "team1Score": 13,
  "team2Score": 3,
  "team1Id": 12064,
  "team2Id": 1120,
  "team1Name": "Nova Esports",
  "team2Name": "EDward Gaming",
  "map": "Split",
  "winner": 12064,
  "playerGameIds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}`,
      },
    ],
  },
  {
    title: 'Map Veto',
    description: 'Ordered ban, pick, and decider actions for each match\'s map selection phase.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/map-veto', auth: true,
        description: 'Paginated list of map veto actions per match. Includes nested team and match objects.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 1,
      "type": "ban",
      "mapSelected": "Bind",
      "turn": 1,
      "match": {
        "team1Id": 12064,
        "team1Name": "Nova Esports",
        "team2Id": 1120,
        "team2Name": "EDward Gaming"
      },
      "team": { "id": 12064, "name": "Nova Esports", "teamTag": "NOVA" }
    }
  ],
  "totalElements": 62001,
  "totalPages": 62001,
  "number": 0,
  "size": 20
}`,
        notes: 'type is one of: "ban", "pick", "decider"',
      },
      {
        method: 'GET', path: '/api/map-veto/{id}', auth: true,
        description: 'Get a single map veto record.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Map veto ID' }],
        responseExample: `{
  "id": 1,
  "type": "ban",
  "mapSelected": "Bind",
  "turn": 1,
  "match": {
    "team1Id": 12064,
    "team1Name": "Nova Esports",
    "team2Id": 1120,
    "team2Name": "EDward Gaming"
  },
  "team": { "id": 12064, "name": "Nova Esports", "teamTag": "NOVA" }
}`,
      },
    ],
  },
  {
    title: 'Tournament Placements',
    description: 'Final standings and prize money allocations for each team in each tournament.',
    icon: <Zap className="w-4 h-4 text-c9-cyan" />,
    endpoints: [
      {
        method: 'GET', path: '/api/tournament-placements', auth: true,
        description: 'Paginated list of team placements per tournament. Includes nested team and tournament objects.',
        params: paginationParams,
        responseExample: `{
  "content": [
    {
      "id": 1,
      "placement": "1",
      "players": null,
      "prizeMoney": null,
      "stage": "main-event",
      "team": { "id": 11328, "name": "FunPlus Phoenix", "teamTag": "FPX" },
      "tournament": { "id": 2720, "name": "China Evolution Series: Epilogue" }
    }
  ],
  "totalElements": 11280,
  "totalPages": 11280,
  "number": 0,
  "size": 20
}`,
      },
      {
        method: 'GET', path: '/api/tournament-placements/{id}', auth: true,
        description: 'Get a single tournament placement.',
        params: [{ name: 'id', in: 'path', type: 'integer', required: true, description: 'Placement ID' }],
        responseExample: `{
  "id": 1,
  "placement": "1",
  "players": null,
  "prizeMoney": null,
  "stage": "main-event",
  "team": { "id": 11328, "name": "FunPlus Phoenix", "teamTag": "FPX" },
  "tournament": { "id": 2720, "name": "China Evolution Series: Epilogue" }
}`,
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
            <EndpointGroup key={g.title} title={g.title} description={g.description} icon={g.icon} endpoints={g.endpoints} />
          ))}
        </div>
      </Card>

    </div>
  );
}
