//import { Film } from 'lucide-react';
import { User } from 'lucide-react';
import { useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import SkyBackground from './components/SkyBackground';
import { TabProps, Tab } from './components/Tab';
import {About} from './pages/About';
import { Home } from './pages/Home';
import { Vods } from './pages/Vods';
import { ApiDocs } from './pages/ApiDocs';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';

const PAGE_ORDER = ['/', '/apidocs', '/vods', '/about'];

const panVariants = {
  enter:  (dir: number) => ({ x: dir >= 0 ? '100%' : '-100%', opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit:   (dir: number) => ({ x: dir >= 0 ? '-100%' : '100%', opacity: 0 }),
};

const panTransition = { duration: 0.6, ease: [0.4, 0, 0.2, 1] as const };

function App() {
  const navData: TabProps[] = [
    { name: "Home", url: "/" },
    { name: "API Docs", url: "/apidocs" },
    { name: "VODs", url: "/vods" },
    { name: "About", url: "/about" }
  ];

  const navigate = useNavigate();
  const location = useLocation();
  const prevPathRef = useRef(location.pathname);

  const prevIdx = PAGE_ORDER.indexOf(prevPathRef.current);
  const currIdx = PAGE_ORDER.indexOf(location.pathname);
  const direction = currIdx >= prevIdx ? 1 : -1;
  prevPathRef.current = location.pathname;

  function accountBtn(){
    navigate("/account");
  }



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
      <nav className="py-10 max-w-7xl mx-auto px-48 flex items-center gap-3 justify-center">
        {navData.map((link) => (
          <Tab
          key={link.url}
          name = {link.name}
          url = {link.url}
          />
        ))}
        <button className = "t-30 h-30 w-30 rounded-2xl bg-c9-cyan p-4 justify-items-center border-2 border-white hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-0.5"
        onClick={accountBtn}>
          <h1 className="text-2xl font-bold tracking-wide">
            <User className="w-8 h-8 text-white" />
          </h1>
        </button>
      </nav>
      <div style={{ position: 'relative', overflow: 'hidden' }}>
        <AnimatePresence mode="popLayout" custom={direction}>
          <motion.div
            key={location.pathname}
            custom={direction}
            variants={panVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={panTransition}
            style={{ willChange: 'transform' }}
          >
            <Routes location={location}>
              <Route path="/" element={<Home />} />
              <Route path="/about" element={<About />} />
              <Route path="/vods" element={<Vods />} />
              <Route path="/apidocs" element={<ApiDocs />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </div>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <></>
      </main>
    </div>
  );
}

export default App;
