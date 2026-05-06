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

export function OngoingTournaments() {
  const [tournaments, setTournaments] = useState<Tournament[]>([]);

  useEffect(() =>{
    async function fetchTournaments(){
      try{
        const response = await fetch('/dashboard/ongoing_tournaments');
        if (!response.ok) {
          console.error("Dashboard API error: ", response.status, response.statusText);
          return;
        }
        const data = await response.json();
        setTournaments(Array.isArray(data) ? data : []);
      } catch (error){
        console.error("Failed to load esports data: ", error);
      }
    }
    fetchTournaments();
    
  }, []);
  
  return (
        <div className = "mt-30 t-30 h-96 max-w-7xl rounded-2xl mx-auto bg-white bg-opacity-55 p-4 justify-items-center border-2 border-c9-cyan hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-1.5 transition duration-1000 ease-in-out"
        >
          <h1 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-cyan font-extrabold">{"Home"}</span>
          </h1>
          <h1>Ongoing VCT Tournaments</h1>
          <ul>
            {tournaments.map((t) => (
              <li key={t.id}>
                {t.name} - {t.location} ({new Date(t.start_date).toLocaleDateString()} - {new Date(t.end_date).toLocaleDateString()})
              </li>
            ))}
          </ul>
        </div>
  );
}