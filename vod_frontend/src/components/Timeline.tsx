import { useState } from 'react';
import { ChevronDown, ChevronRight, Skull, Trophy, Clock } from 'lucide-react';
import { TimelineData, RoundData, KillEvent } from '../api';

interface TimelineProps {
  data?: TimelineData | null;
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function KillRow({ kill, index }: { kill: KillEvent; index: number }) {
  // Use team color classes based on killer/victim team color
  const getTeamColorClass = (teamColor: string) => {
    if (teamColor === 'teal' || teamColor === 'cyan' || teamColor === 'green') {
      return 'text-valorant-teal';
    }
    if (teamColor === 'orange' || teamColor === 'red') {
      return 'text-valorant-orange';
    }
    return 'text-gray-400';
  };

  const killerColor = getTeamColorClass(kill.killer_team);
  const victimColor = getTeamColorClass(kill.victim_team);

  return (
    <div className="flex items-center gap-3 py-2 px-3 hover:bg-gray-800/50 rounded transition-colors">
      <span className="text-gray-500 w-6 text-right text-sm">{index + 1}</span>
      <Clock className="w-4 h-4 text-gray-500" />
      <span className="text-gray-400 text-sm w-16">{kill.timestamp_display}</span>
      <span className={`font-medium ${killerColor}`}>{kill.killer_name}</span>
      <Skull className="w-4 h-4 text-red-500 mx-1" />
      <span className={`font-medium ${victimColor}`}>{kill.victim_name}</span>
      {kill.is_headshot && (
        <span className="ml-2 px-2 py-0.5 bg-yellow-500/20 text-yellow-500 text-xs rounded">
          HS
        </span>
      )}
      {kill.weapon !== 'unknown' && (
        <span className="ml-auto text-gray-500 text-sm">{kill.weapon}</span>
      )}
    </div>
  );
}

function RoundCard({ round, leftTeamName, rightTeamName }: { 
  round: RoundData; 
  leftTeamName?: string;
  rightTeamName?: string;
}) {
  const safeRoundNumber = typeof round.round_number === 'number' ? round.round_number : 0;
  const [isExpanded, setIsExpanded] = useState(safeRoundNumber <= 3);
  
  // Determine winner based on score change or winner field
  const leftScore = round.score?.left ?? 0;
  const rightScore = round.score?.right ?? 0;
  const leftTeamCode = round.score?.left_team || leftTeamName || 'Left';
  const rightTeamCode = round.score?.right_team || rightTeamName || 'Right';
  
  // Winner is determined by which team's score increased this round
  const winnerDisplay = round.winner === 'teal' || round.winner === leftTeamCode
    ? leftTeamCode
    : round.winner === 'orange' || round.winner === rightTeamCode
      ? rightTeamCode
      : null;
  
  const winnerColor = (round.winner === 'teal' || round.winner === leftTeamCode) 
    ? 'text-valorant-teal' 
    : 'text-valorant-orange';
  const kills = Array.isArray(round.kills) ? round.kills : [];

  return (
    <div className="bg-valorant-gray rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          {isExpanded ? (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-400" />
          )}
          <span className="font-bold text-lg">Round {round.round_number}</span>
          <span className="text-gray-400 text-sm">
            (<span className="text-valorant-teal">{leftTeamCode}</span>
            {' '}
            <span className="font-medium">{leftScore}</span>
            {' - '}
            <span className="font-medium">{rightScore}</span>
            {' '}
            <span className="text-valorant-orange">{rightTeamCode}</span>)
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Skull className="w-4 h-4 text-gray-500" />
            <span className="text-gray-400">{kills.length} kills</span>
          </div>
          {winnerDisplay && (
            <div className="flex items-center gap-2">
              <Trophy className={`w-4 h-4 ${winnerColor}`} />
              <span className={`font-medium ${winnerColor}`}>{winnerDisplay}</span>
            </div>
          )}
          <span className="text-gray-500 text-sm">
            {formatTime(round.start_ms || 0)} - {formatTime(round.end_ms || 0)}
          </span>
        </div>
      </button>
      
      {isExpanded && (
        <div className="border-t border-gray-700 px-4 py-2">
          {kills.length > 0 ? (
            <div className="divide-y divide-gray-700/50">
              {kills.map((kill, idx) => (
                <KillRow key={idx} kill={kill} index={idx} />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 py-4 text-center">No kills detected in this round</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function Timeline({ data }: TimelineProps) {
  if (!data) return null;

  const leftTeamName = data.match_info?.left_team || 'Left Team';
  const rightTeamName = data.match_info?.right_team || 'Right Team';

  const rounds = Array.isArray(data.rounds) ? data.rounds : [];

  // Get final score from the last round
  const lastRound = rounds.length > 0 ? rounds[rounds.length - 1] : null;
  const finalLeftScore = lastRound?.score?.left ?? 0;
  const finalRightScore = lastRound?.score?.right ?? 0;

  // Calculate kill stats by team code (not color)
  const teamStats = rounds.reduce(
    (acc, round) => {
      const kills = Array.isArray(round?.kills) ? round.kills : [];
      kills.forEach(kill => {
        // Count by team color for kills
        if (kill && (kill.killer_team === 'teal' || kill.killer_team === 'cyan')) {
          acc.leftKills++;
        } else if (kill && (kill.killer_team === 'orange' || kill.killer_team === 'red')) {
          acc.rightKills++;
        }
      });
      
      return acc;
    },
    { leftKills: 0, rightKills: 0 }
  );

  return (
    <div className="space-y-6">
      {/* Match Summary */}
      <div className="bg-valorant-gray rounded-xl p-6">
        <div className="grid grid-cols-3 gap-8">
          {/* Left Team */}
          <div className="text-center">
            <p className="text-valorant-teal font-bold text-2xl mb-1">{leftTeamName}</p>
            <p className="text-4xl font-bold">{finalLeftScore}</p>
            <p className="text-gray-500 text-sm mt-1">{teamStats.leftKills} kills</p>
          </div>
          
          {/* Match Info */}
          <div className="text-center">
            <p className="text-gray-400 text-sm mb-2">
              {data.match_info?.map || 'Unknown Map'}
            </p>
            <p className="text-2xl font-bold">
              <span className="text-valorant-teal">{finalLeftScore}</span>
              <span className="text-gray-500 mx-2">-</span>
              <span className="text-valorant-orange">{finalRightScore}</span>
            </p>
            <p className="text-gray-500 text-sm mt-2">
              {data.total_kills ?? rounds.reduce((s, r) => s + ((r && Array.isArray(r.kills)) ? r.kills.length : 0), 0)} total kills • {rounds.length} rounds
            </p>
          </div>
          
          {/* Right Team */}
          <div className="text-center">
            <p className="text-valorant-orange font-bold text-2xl mb-1">{rightTeamName}</p>
            <p className="text-4xl font-bold">{finalRightScore}</p>
            <p className="text-gray-500 text-sm mt-1">{teamStats.rightKills} kills</p>
          </div>
        </div>
      </div>

      {/* Rounds */}
      <div className="space-y-3">
        <h3 className="text-lg font-bold text-gray-300 mb-4">Round by Round</h3>
        {rounds.map((round) => (
          <RoundCard 
            key={round?.round_number ?? Math.random()} 
            round={round as RoundData} 
            leftTeamName={leftTeamName}
            rightTeamName={rightTeamName}
          />
        ))}
      </div>
    </div>
  );
}
