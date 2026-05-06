import { OngoingTournaments } from '../components/dashboard_components/OngoingTournaments';
import { RecentMatches } from '../components/dashboard_components/RecentMatches';
import { PlayerStats } from '../components/dashboard_components/PlayerStats';
import { TournamentMapStats } from '../components/dashboard_components/TournamentMapStats';
import { AgentPickRates } from '../components/dashboard_components/AgentPickRates';

export function Home() {
  return (
    <>
      <OngoingTournaments/>
      <RecentMatches/>
      <PlayerStats/>
      <TournamentMapStats/>
      <AgentPickRates/>
    </>
  );
}