export function About() {
  // ── Team members ─────────────────────────────────────────
  // Add or remove entries here. The initials are derived from the name.
  const team: { name: string; role: string }[] = [
    { name: 'Name Here', role: 'Role / Title' },
    { name: 'Name Here', role: 'Role / Title' },
    { name: 'Name Here', role: 'Role / Title' },
  ];

  // ── Tech stack tags ───────────────────────────────────────
  const stack = ['React', 'TypeScript', 'Spring Boot', 'PostgreSQL', 'Docker'];

  return (
    <div className="p-4 px-48 max-w-7xl mx-auto space-y-6">

      {/* ── Hero ──────────────────────────────────────────── */}
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-10 text-center">
        <h1 className="text-5xl font-extrabold text-c9-cyan mb-4 tracking-wide">
          {/* Replace with your project / team name */}
          About C9 VOD Processor
        </h1>
        <p className="text-c9-muted text-xl max-w-2xl mx-auto leading-relaxed">
          {/* Replace with a one-liner about what this project is and who it's for */}
          A brief description of what this project does and who built it.
        </p>
      </div>

      {/* ── Mission ───────────────────────────────────────── */}
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-8">
        <h2 className="text-2xl font-bold text-c9-text mb-3">Our Mission</h2>
        <p className="text-c9-muted leading-relaxed">
          {/* Replace with your mission statement */}
          Describe the goal of the project here — what problem it solves and why it matters.
        </p>
      </div>

      {/* ── Team ──────────────────────────────────────────── */}
      <div className="bg-white/80 backdrop-blur-md rounded-3xl border-2 border-c9-cyan/40 shadow-sm p-8">
        <h2 className="text-2xl font-bold text-c9-text mb-6">The Team</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {team.map((member) => {
            const initials = member.name
              .split(' ')
              .map((w) => w[0])
              .join('')
              .toUpperCase();
            return (
              <div
                key={member.name + member.role}
                className="bg-c9-bg rounded-2xl border border-c9-cyan/20 p-5 text-center"
              >
                <div className="w-16 h-16 rounded-full bg-c9-cyan/20 border-2 border-c9-cyan/30 mx-auto mb-3 flex items-center justify-center">
                  <span className="text-c9-cyan text-xl font-extrabold">{initials}</span>
                </div>
                <h3 className="text-c9-text font-semibold">{member.name}</h3>
                <p className="text-c9-muted text-sm">{member.role}</p>
              </div>
            );
          })}
        </div>
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
