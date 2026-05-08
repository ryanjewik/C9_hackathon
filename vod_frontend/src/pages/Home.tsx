import { OngoingTournaments } from '../components/dashboard_components/OngoingTournaments';
import { RecentMatches } from '../components/dashboard_components/RecentMatches';
import { PlayerStats } from '../components/dashboard_components/PlayerStats';
import { TournamentMapStats } from '../components/dashboard_components/TournamentMapStats';
import { AgentPickRates } from '../components/dashboard_components/AgentPickRates';

export function Home() {
  return (
    <div className="p-4 px-48 max-w-7xl mx-auto space-y-4">

      {/* Row 1: Live context — full width compact banner */}
      <div>
        <OngoingTournaments />
      </div>

      {/* Row 2: Recent Matches — full width */}
      <div>
        <RecentMatches />
      </div>

      {/* Row 3: Player stats (wide left) + Agent picks + Map stats stacked (narrow right) — equal height */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-stretch">
        <div className="lg:col-span-2 flex flex-col">
          <PlayerStats />
        </div>
        <div className="lg:col-span-1 flex flex-col gap-4">
          <div className="flex-1 flex flex-col">
            <AgentPickRates />
          </div>
          <div className="flex-1 flex flex-col">
            <TournamentMapStats />
          </div>
        </div>
      </div>

    </div>
  );
}