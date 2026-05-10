import { Film } from 'lucide-react';

export function Vods() {
  return (
    <div className="flex items-center justify-center min-h-[60vh] px-6">
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-xl p-16 text-center max-w-lg w-full">
        <div className="w-20 h-20 rounded-full bg-c9-cyan/15 border-2 border-c9-cyan/30 mx-auto mb-6 flex items-center justify-center">
          <Film className="w-10 h-10 text-c9-cyan" />
        </div>
        <h1 className="text-4xl font-extrabold text-c9-cyan mb-3 tracking-wide">
          Coming Soon
        </h1>
        <p className="text-c9-muted text-lg leading-relaxed">
          VOD uploads and replay analysis are on the way.
          <br />
          Check back soon!
        </p>
      </div>
    </div>
  );
}
