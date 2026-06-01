//import { Film } from 'lucide-react';
import { User } from 'lucide-react';
import { useRef, useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import SkyBackground from './components/SkyBackground';
import { TabProps, Tab } from './components/Tab';
import { AuthModal } from './components/AuthModal';
import {About} from './pages/About';
import { Home } from './pages/Home';
import { Vods } from './pages/Vods';
import { ApiDocs } from './pages/ApiDocs';
import { Routes, Route, useLocation } from 'react-router-dom';

const PAGE_ORDER = ['/', '/apidocs', '/vods', '/about'];

const panVariants = {
  enter:  (dir: number) => ({ x: dir >= 0 ? '100%' : '-100%', opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit:   (dir: number) => ({ x: dir >= 0 ? '-100%' : '100%', opacity: 0 }),
};

const panTransition = { duration: 0.6, ease: [0.4, 0, 0.2, 1] as const };

const verticalTransition = { duration: 0.55, ease: [0.4, 0, 0.2, 1] as const };

function App() {
  const navData: TabProps[] = [
    { name: "Home", url: "/" },
    { name: "API Docs", url: "/apidocs" },
    { name: "VODs", url: "/vods" },
    { name: "About", url: "/about" }
  ];

  const location = useLocation();
  const prevPathRef = useRef(location.pathname);

  const prevIdx = PAGE_ORDER.indexOf(prevPathRef.current);
  const currIdx = PAGE_ORDER.indexOf(location.pathname);
  const direction = currIdx >= prevIdx ? 1 : -1;
  prevPathRef.current = location.pathname;

  // Auth state
  const [authOpen, setAuthOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<string | null>(() =>
    localStorage.getItem('c9_username')
  );

  useEffect(() => {
    if (currentUser) {
      localStorage.setItem('c9_username', currentUser);
    } else {
      localStorage.removeItem('c9_username');
      localStorage.removeItem('c9_token');
    }
  }, [currentUser]);

  function handleAuthSuccess(token: string, username: string) {
    localStorage.setItem('c9_token', token);
    setCurrentUser(username);
    setAuthOpen(false);
  }

  function handleLogout() {
    setCurrentUser(null);
    setAuthOpen(false);
  }

  return (
    <div className="min-h-screen" style={{ position: 'relative', overflow: 'hidden' }}>
      <SkyBackground />

      <AnimatePresence mode="wait">
        {authOpen ? (
          /* ── Auth modal pans up from below ── */
          <motion.div
            key="auth"
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={verticalTransition}
            style={{ willChange: 'transform', position: 'relative', minHeight: '100vh' }}
          >
            <AuthModal
              onClose={() => setAuthOpen(false)}
              onSuccess={handleAuthSuccess}
              currentUser={currentUser}
              onLogout={handleLogout}
            />
          </motion.div>
        ) : (
          /* ── Main content exits downward when auth opens ── */
          <motion.div
            key="main"
            initial={{ y: 0 }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={verticalTransition}
            style={{ willChange: 'transform' }}
          >
            <nav className="py-10 max-w-7xl mx-auto px-48 flex items-center gap-3 justify-center">
              {navData.map((link) => (
                <Tab
                  key={link.url}
                  name={link.name}
                  url={link.url}
                />
              ))}
              <button
                className="t-30 h-30 w-30 rounded-2xl bg-c9-cyan p-4 justify-items-center border-2 border-white hover:shadow-lg hover:translate-x-0.4 hover:-translate-y-0.5"
                onClick={() => setAuthOpen(true)}
              >
                {currentUser ? (
                  <span className="text-white text-sm font-bold leading-none">
                    {currentUser.slice(0, 2).toUpperCase()}
                  </span>
                ) : (
                  <User className="w-8 h-8 text-white" />
                )}
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
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
