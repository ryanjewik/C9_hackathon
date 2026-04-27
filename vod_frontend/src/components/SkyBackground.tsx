interface CloudDef {
  top: string;
  duration: string;
  delay: string;
  scale: number;
  opacity: number;
  shape: 'fluffy' | 'wide' | 'small';
}

const CLOUDS: CloudDef[] = [
  { top: '4%',  duration: '90s',  delay: '0s',    scale: 1.4, opacity: 0.82, shape: 'fluffy' },
  { top: '55%', duration: '115s', delay: '-40s',  scale: 1.2, opacity: 0.68, shape: 'wide'   },
  { top: '2%',  duration: '75s',  delay: '-62s',  scale: 1.5, opacity: 0.70, shape: 'small'  },
  { top: '68%', duration: '100s', delay: '-25s',  scale: 1.1, opacity: 0.58, shape: 'wide'   },
  { top: '28%', duration: '130s', delay: '-80s',  scale: 1.3, opacity: 0.55, shape: 'fluffy' },
  { top: '8%',  duration: '85s',  delay: '-50s',  scale: 1.0, opacity: 0.72, shape: 'wide'   },
  { top: '78%', duration: '110s', delay: '-15s',  scale: 1.2, opacity: 0.55, shape: 'small'  },
  { top: '45%', duration: '95s',  delay: '-70s',  scale: 1.1, opacity: 0.65, shape: 'fluffy' },
];

// Cloud shapes built from stacked ellipses/circles for a natural fluffy look
// overflow="visible" lets edge ellipses curve freely past the SVG bounds.
// The scale wrapper below uses clip-path inset to clip top+bottom only.
// Edge ellipses are always smaller (rx, ry) than their inner neighbours.
const CLOUD_PATHS: Record<CloudDef['shape'], JSX.Element> = {
  // 4 bumps — asymmetric, peak at 3rd bump; edges taper clearly
  fluffy: (
    <svg viewBox="0 0 320 140" overflow="visible" xmlns="http://www.w3.org/2000/svg" style={{ width: 320, height: 140 }}>
      <ellipse cx="52"  cy="118" rx="50" ry="56" />
      <ellipse cx="118" cy="106" rx="64" ry="66" />
      <ellipse cx="196" cy="96"  rx="72" ry="72" />
      <ellipse cx="266" cy="110" rx="54" ry="58" />
    </svg>
  ),
  // 3 bumps — wide spacing, tallest in middle; edges taper down on both sides
  wide: (
    <svg viewBox="0 0 300 136" overflow="visible" xmlns="http://www.w3.org/2000/svg" style={{ width: 300, height: 136 }}>
      <ellipse cx="62"  cy="108" rx="54" ry="56" />
      <ellipse cx="158" cy="94"  rx="66" ry="70" />
      <ellipse cx="250" cy="110" rx="46" ry="52" />
    </svg>
  ),
  // 4 bumps — compact, middle two are tallest; edges noticeably smaller
  small: (
    <svg viewBox="0 0 280 136" overflow="visible" xmlns="http://www.w3.org/2000/svg" style={{ width: 280, height: 136 }}>
      <ellipse cx="46"  cy="114" rx="44" ry="52" />
      <ellipse cx="110" cy="98"  rx="62" ry="66" />
      <ellipse cx="182" cy="100" rx="60" ry="64" />
      <ellipse cx="244" cy="114" rx="42" ry="50" />
    </svg>
  ),
};

export default function SkyBackground() {
  return (
    <div className="sky-bg" aria-hidden="true">
      {/* Blob accents */}
      <div className="sky-blob sky-blob-aqua" />
      <div className="sky-blob sky-blob-rose" />
      <div className="sky-blob sky-blob-accent" />

      {/* Drifting SVG clouds — outer div animates translateX, inner div scales */}
      {CLOUDS.map((cloud, i) => (
        <div
          key={i}
          className="sky-cloud-wrapper"
          style={{
            top: cloud.top,
            animationDuration: cloud.duration,
            animationDelay: cloud.delay,
            opacity: cloud.opacity,
          }}
        >
          <div
            style={{
              transform: `scale(${cloud.scale})`,
              transformOrigin: 'left center',
              // clip only top/bottom; negative left/right allows ellipses to bleed past SVG edges
              clipPath: 'inset(0 -9000px 0 -9000px)',
            }}
          >
            <div className="sky-cloud-svg">
              {CLOUD_PATHS[cloud.shape]}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}


