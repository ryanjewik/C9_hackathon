import {useState, useEffect} from 'react';

interface Tournament {
  id: number;
  name: string;
  tier: string;
  start_date: string;
  end_date: string;
  location: string;
  prize_pool: string;
  status: string;
}

function formatDate(d: string | null) {
  if (!d) return '—';
  return new Date(d).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function daysLeft(end: string | null) {
  if (!end) return null;
  const diff = Math.ceil((new Date(end).getTime() - Date.now()) / 86400000);
  return diff > 0 ? diff : 0;
}

export function OngoingTournaments() {
  const [tournaments, setTournaments] = useState<Tournament[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStart = Date.now();
    async function fetchTournaments() {
      try {
        const response = await fetch('/dashboard/ongoing_tournaments');
        if (!response.ok) return;
        const data = await response.json();
        setTournaments(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error('Failed to load tournaments:', error);
      } finally {
        const elapsed = Date.now() - fetchStart;
        setTimeout(() => setLoading(false), Math.max(0, 650 - elapsed));
      }
    }
    fetchTournaments();
  }, []);

  return (
    <div className="h-full bg-white bg-opacity-55 rounded-2xl p-4 border-2 border-c9-cyan hover:shadow-lg hover:-translate-y-1.5 transition duration-1000 ease-in-out">
      <h2 className="text-2xl font-bold tracking-wide mb-4">
        <span className="text-c9-cyan font-extrabold">Ongoing Tournaments</span>
      </h2>

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 animate-pulse">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="rounded-xl border border-c9-cyan/30 bg-white/60 p-4 flex flex-col gap-3">
              <div className="h-3 w-16 bg-c9-cyan/30 rounded-full" />
              <div className="h-4 w-3/4 bg-gray-200 rounded" />
              <div className="h-3 w-1/2 bg-gray-100 rounded" />
              <div className="h-3 w-2/3 bg-gray-100 rounded" />
            </div>
          ))}
        </div>
      ) : tournaments.length === 0 ? (
        <p className="text-c9-muted text-sm text-center py-8">No ongoing tournaments</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {tournaments.map((t) => {
            const days = daysLeft(t.end_date);
            return (
              <div
                key={t.id}
                className="flex flex-col gap-2 rounded-xl border border-c9-cyan bg-white bg-opacity-70 p-4 hover:shadow-md transition"
              >
                {/* Tier badge + live pill */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full bg-c9-cyan text-white">
                    {t.tier}
                  </span>
                  <span className="flex items-center gap-1 text-[10px] font-semibold text-red-500 uppercase tracking-wider">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                    Live
                  </span>
                </div>

                {/* Name */}
                <p className="text-sm font-bold text-c9-text leading-snug">{t.name}</p>

                {/* Location */}
                {t.location && (
                  <p className="text-xs text-c9-muted flex items-start gap-1">
                    <span>📍</span>
                    <span>{t.location}</span>
                  </p>
                )}

                {/* Dates */}
                <p className="text-xs text-c9-muted">
                  {formatDate(t.start_date)} — {formatDate(t.end_date)}
                </p>

                {/* Days left */}
                {days !== null && (
                  <p className="text-xs font-semibold" style={{ color: days <= 3 ? '#e74c3c' : '#1abc9c' }}>
                    {days === 0 ? 'Ends today' : `${days} day${days === 1 ? '' : 's'} left`}
                  </p>
                )}

                {/* Prize pool */}
                {t.prize_pool && t.prize_pool !== 'TBD' && t.prize_pool !== '$0 USD' && (
                  <p className="text-xs text-c9-muted mt-auto pt-1 border-t border-gray-100">
                    💰 {t.prize_pool.split('~')[0].trim()}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}