import { Github, Globe, Linkedin } from 'lucide-react';

export function About() {
  const stack = ['React', 'TypeScript', 'Java', 'Spring Boot', 'PostgreSQL', 'Python', 'FastAPI', 'Redis', 'Docker', 'LGTM Stack'];

  return (
    <div className="p-4 px-48 max-w-7xl mx-auto space-y-6">
      {/* ── Creator ───────────────────────────────────────── */}
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-8">
        <h2 className="text-2xl font-bold text-c9-text mb-6">About the Creator</h2>
        <div className="flex flex-col items-center text-center gap-4">
          <div className="w-20 h-20 rounded-full bg-c9-cyan/20 border-2 border-c9-cyan/30 flex items-center justify-center">
            <span className="text-c9-cyan text-2xl font-extrabold">RJ</span>
          </div>
          <div>
            <h3 className="text-c9-text font-semibold text-lg">Ryan Jewik</h3>
            <p className="text-c9-muted text-sm">Software Engineer</p>
          </div>
          <div className="flex gap-4">
            <a
              href="https://ryanhideo.dev/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-c9-cyan/10 border border-c9-cyan/30 text-c9-cyan font-semibold rounded-xl text-sm hover:bg-c9-cyan/20 transition"
            >
              <Globe className="w-4 h-4" /> Portfolio
            </a>
            <a
              href="https://github.com/ryanjewik"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-c9-cyan/10 border border-c9-cyan/30 text-c9-cyan font-semibold rounded-xl text-sm hover:bg-c9-cyan/20 transition"
            >
              <Github className="w-4 h-4" /> GitHub
            </a>
            <a
              href="https://www.linkedin.com/in/ryanjewik/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-c9-cyan/10 border border-c9-cyan/30 text-c9-cyan font-semibold rounded-xl text-sm hover:bg-c9-cyan/20 transition"
            >
              <Linkedin className="w-4 h-4" /> LinkedIn
            </a>
          </div>
        </div>
      </div>

      {/* ── Inspiration ───────────────────────────────────── */}
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-8">
        <h2 className="text-2xl font-bold text-c9-text mb-3">Inspiration</h2>
        <p className="text-c9-muted leading-relaxed">
        While searching through different VCT data apis, I noticed that most did not have per map agent data for players. This I felt was especially important if you wanted to answer questions like a player's performance between different agents, or which players performed the best on a specific agent. Hence why this api exists, to fill in the gap. As I continued to build this app more ideas spawned in and I began working on a computer vision model to help speed up the VOD review process for professional matches. Now what I want this to be is a VALORANT analytics platform where both professionals players/coaches, developers, and viewers can gain additional insights on team macro, timings, tendencies, and even perform their own analytics!
        </p>
      </div>

      {/* ── About the Website ─────────────────────────────── */}
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-8">
        <h2 className="text-2xl font-bold text-c9-text mb-3">About the Website</h2>
        <p className="text-c9-muted leading-relaxed">
          This platform as of right now I plan for two main components. The API for developers and analysts to use for their own analytics, and the VOD processor for professionals and viewers alike.
          The API stands out with it's inclusion of agents-played-per-map-per-player. Other APIs failed to include agent data for each map, only for entire matches.
          The VOD processor will allow for additional insights on VOD review, including maps of player positions for kills/deaths, a timeline of killfeed events, and additional insights on a match VOD. For now I am building one that processes VCT matches, but will eventually build a slimmer version for individual players to pass their own POVs. 
          
          Please have a look at the apidocs for how to use the data, and the home page serves as a demonstration for what you can do with the data!
          I hope you all enjoy what I have been building thus far, I have some other ideas for the platform and different ways of interfacing with the data planned but for now these features take priority. 
          If you have any questions or requests I encourage you to reach out via my portfolio website's contact page or via LinkedIn!
        </p>
      </div>

      {/* ── Tech Stack ────────────────────────────────────── */}
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-8">
        <h2 className="text-2xl font-bold text-c9-text mb-6">Tech Stack</h2>
        <div className="flex flex-wrap gap-3">
          {stack.map((tech) => (
            <span
              key={tech}
              className="px-4 py-2 bg-c9-cyan/10 border border-c9-cyan/30 text-c9-cyan font-semibold rounded-xl text-sm"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>

    </div>
  );
}
