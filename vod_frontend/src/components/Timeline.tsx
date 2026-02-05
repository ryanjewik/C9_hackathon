import { useState } from 'react';
import { ChevronDown, ChevronRight, Skull, Trophy, Clock } from 'lucide-react';
import { TimelineData, RoundData, KillEvent } from '../api';

interface TimelineProps {
  data: TimelineData;
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function KillRow({ kill, index }: { kill: KillEvent; index: number }) {
  const killerColor = kill.killer_team === 'teal' ? 'text-valorant-teal' : 'text-valorant-orange';
  const victimColor = kill.victim_team === 'teal' ? 'text-valorant-teal' : 'text-valorant-orange';

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
  const [isExpanded, setIsExpanded] = useState(round.round_number <= 3);
  
  const winnerDisplay = round.winner === 'teal' 
    ? leftTeamName || 'Left Team'
    : round.winner === 'orange' 
      ? rightTeamName || 'Right Team'
      : 'Unknown';
  
  const winnerColor = round.winner === 'teal' ? 'text-valorant-teal' : 'text-valorant-orange';

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
          {round.score && (
            <span className="text-gray-400 text-sm">
              ({round.score.left_team} - {round.score.right_team})
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Skull className="w-4 h-4 text-gray-500" />
            <span className="text-gray-400">{round.kills.length} kills</span>
          </div>
          {round.winner && (
            <div className="flex items-center gap-2">
              <Trophy className={`w-4 h-4 ${winnerColor}`} />
              <span className={`font-medium ${winnerColor}`}>{winnerDisplay}</span>
            </div>
          )}
          <span className="text-gray-500 text-sm">
            {formatTime(round.start_ms)} - {formatTime(round.end_ms)}
          </span>
        </div>
      </button>
      
      {isExpanded && (
        <div className="border-t border-gray-700 px-4 py-2">
          {round.kills.length > 0 ? (
            <div className="divide-y divide-gray-700/50">
              {round.kills.map((kill, idx) => (
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
  const leftTeamName = data.match_info?.left_team || 'Left Team';
  const rightTeamName = data.match_info?.right_team || 'Right Team';

  // Calculate team stats
  const teamStats = data.rounds.reduce(
    (acc, round) => {
      if (round.winner === 'teal') acc.leftWins++;
      else if (round.winner === 'orange') acc.rightWins++;
      
      round.kills.forEach(kill => {
        if (kill.killer_team === 'teal') acc.leftKills++;
        else if (kill.killer_team === 'orange') acc.rightKills++;
      });
      
      return acc;
    },
    { leftWins: 0, rightWins: 0, leftKills: 0, rightKills: 0 }
  );

  return (
    <div className="space-y-6">
      {/* Match Summary */}
      <div className="bg-valorant-gray rounded-xl p-6">
        <div className="grid grid-cols-3 gap-8">
          {/* Left Team */}
          <div className="text-center">
            <p className="text-valorant-teal font-bold text-2xl mb-1">{leftTeamName}</p>
            <p className="text-4xl font-bold">{teamStats.leftWins}</p>
            <p className="text-gray-500 text-sm mt-1">{teamStats.leftKills} kills</p>
          </div>
          
          {/* Match Info */}
          <div className="text-center">
            <p className="text-gray-400 text-sm mb-2">
              {data.match_info?.map || 'Unknown Map'}
            </p>
            <p className="text-2xl font-bold">
              <span className="text-valorant-teal">{teamStats.leftWins}</span>
              <span className="text-gray-500 mx-2">-</span>
              <span className="text-valorant-orange">{teamStats.rightWins}</span>
            </p>
            <p className="text-gray-500 text-sm mt-2">
              {data.total_kills} total kills • {data.rounds.length} rounds
            </p>
          </div>
          
          {/* Right Team */}
          <div className="text-center">
            <p className="text-valorant-orange font-bold text-2xl mb-1">{rightTeamName}</p>
            <p className="text-4xl font-bold">{teamStats.rightWins}</p>
            <p className="text-gray-500 text-sm mt-1">{teamStats.rightKills} kills</p>
          </div>
        </div>
      </div>

      {/* Rounds */}
      <div className="space-y-3">
        <h3 className="text-lg font-bold text-gray-300 mb-4">Round by Round</h3>
        {data.rounds.map((round) => (
          <RoundCard 
            key={round.round_number} 
            round={round} 
            leftTeamName={leftTeamName}
            rightTeamName={rightTeamName}
          />
        ))}
      </div>
    </div>
  );
}
