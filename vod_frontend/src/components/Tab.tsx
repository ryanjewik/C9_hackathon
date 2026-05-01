import { useNavigate } from 'react-router-dom';

export interface TabProps {
    name: string;
    url: string;
}

export function Tab({name, url}: TabProps){
    const navigate = useNavigate();

    function handleClick(){
        navigate(url);
    }

    return (
        <div className = "t-30 h-30 flex-1 rounded-2xl bg-white p-4 justify-items-center border-2 border-c9-cyan hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-1.5 transition duration-1000 ease-in-out"
        onClick={handleClick}>
          <h1 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-cyan font-extrabold">{name}</span>
          </h1>
        </div>
    );
}