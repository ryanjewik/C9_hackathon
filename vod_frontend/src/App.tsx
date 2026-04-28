//import { Film } from 'lucide-react';
import SkyBackground from './components/SkyBackground';

function App() {
  return (
    <div className="min-h-screen">
      <SkyBackground />
      {/* Header */}
      {/* <header className="bg-white/70 backdrop-blur-md border-b border-c9-cyan/30 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <Film className="w-8 h-8 text-c9-cyan" />
          <h1 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-cyan font-extrabold">C9</span>
            <span className="text-c9-text"> VOD</span>
            <span className="text-c9-muted font-light"> Processor</span>
          </h1>
          <span className="text-c9-muted text-sm ml-2 tracking-widest uppercase">Cloud9 · Timeline Extractor</span>
        </div>
      </header> */}
      <div className="pt-10 max-w-7xl mx-auto flex items-center gap-3 justify-center">
        <div className = "t-30 h-30 w-40 rounded-2xl bg-white p-4 justify-items-center border-2 border-c9-cyan hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-1.5 transition duration-1.5 ease-in-out"
        >
          <h1 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-cyan font-extrabold">C9</span>
          </h1>
        </div>
        <div className = "t-30 h-30 w-40 rounded-2xl bg-white p-4 border border-c9-cyan transition hover:animate-bounce hover:[animation-iteration-count:1]">
          <h1 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-text"> C9</span>
          </h1>
        </div>
        <div className = "t-30 h-30 w-40 rounded-2xl bg-white p-4 border border-c9-cyan">
          <h1 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-muted font-light"> C9</span>
          </h1>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <></>
      </main>
    </div>
  );
}

export default App;
