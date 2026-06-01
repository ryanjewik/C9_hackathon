import { useState, useEffect, useCallback } from 'react';
import {
  X, User, Mail, Lock, LogOut, Users, Key, Bell, Plus,
  ChevronDown, ChevronRight, Trash2, Shield, Crown, Check, XCircle,
  RefreshCw, Copy, UserPlus, AlertTriangle,
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface TeamData {
  id: string;
  name: string;
  ownerUserId: string;
  createdAt: string;
}

interface MemberData {
  userId: string;
  username: string;
  role: string;
  joinedAt: string;
}

interface ApiKeyData {
  id: string;
  name: string;
  keyPrefix: string;
  createdAt: string;
  key?: string;
}

interface InviteData {
  id: string;
  sendingTeam: string;
  sendingAdmin: string;
  createdAt: string;
  teamName?: string;
}

interface ProfileData {
  id: string;
  username: string;
  email: string;
  createdAt: string;
}

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  onClose: () => void;
  onSuccess: (token: string, username: string) => void;
  currentUser: string | null;
  onLogout: () => void;
}

type AuthTab = 'login' | 'register';
type PanelTab = 'profile' | 'teams' | 'invites';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function authHeaders() {
  const token = localStorage.getItem('c9_token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

function friendlyError(data: Record<string, string>, status: number, context?: string): string {
  const msg = data.message ?? '';
  if (context === 'login') {
    if (status === 401 || msg === 'invalid_credentials') return 'Invalid username or password.';
    if (status === 404) return 'No account found with that username.';
  }
  if (context === 'register') {
    if (msg === 'username_taken') return 'That username is already taken.';
    if (msg === 'email_taken') return 'That email is already registered.';
    if (msg === 'username_email_password_required') return 'All fields are required.';
  }
  if (context === 'profile') {
    if (msg === 'username_taken') return 'That username is already taken.';
    if (msg === 'email_taken') return 'That email is already registered.';
    if (msg === 'current_password_required') return 'Enter your current password to set a new one.';
    if (msg === 'invalid_current_password') return 'Current password is incorrect.';
  }
  return data.error ?? data.message ?? 'Something went wrong.';
}

function RoleBadge({ role }: { role: string }) {
  const styles: Record<string, string> = {
    owner: 'bg-amber-100 text-amber-700 border border-amber-300',
    admin: 'bg-blue-100 text-blue-700 border border-blue-300',
    member: 'bg-gray-100 text-gray-600 border border-gray-300',
  };
  const icons: Record<string, JSX.Element> = {
    owner: <Crown className="w-3 h-3" />,
    admin: <Shield className="w-3 h-3" />,
    member: <User className="w-3 h-3" />,
  };
  const r = role?.toLowerCase() ?? 'member';
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${styles[r] ?? styles.member}`}>
      {icons[r] ?? icons.member}
      {r.charAt(0).toUpperCase() + r.slice(1)}
    </span>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function AuthModal({ onClose, onSuccess, currentUser, onLogout }: Props) {
  // ── Auth form state ──
  const [authTab, setAuthTab] = useState<AuthTab>('login');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  // ── Panel state (logged-in) ──
  const [panelTab, setPanelTab] = useState<PanelTab>('profile');

  // Profile
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [newUsername, setNewUsername] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [profileMsg, setProfileMsg] = useState<{ text: string; ok: boolean } | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);

  // Teams
  const [teams, setTeams] = useState<TeamData[]>([]);
  const [teamsLoading, setTeamsLoading] = useState(false);
  const [expandedTeam, setExpandedTeam] = useState<string | null>(null);
  const [teamMembers, setTeamMembers] = useState<Record<string, MemberData[]>>({});
  const [teamKeys, setTeamKeys] = useState<Record<string, ApiKeyData[]>>({});
  const [newTeamName, setNewTeamName] = useState('');
  const [newTeamNameError, setNewTeamNameError] = useState('');
  const [newKeyName, setNewKeyName] = useState<Record<string, string>>({});
  const [revealedKey, setRevealedKey] = useState<Record<string, string>>({});
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [teamMsg, setTeamMsg] = useState<{ text: string; ok: boolean } | null>(null);
  // Per-team invite
  const [inviteQuery, setInviteQuery] = useState<Record<string, string>>({});
  const [inviteSearchMsg, setInviteSearchMsg] = useState<Record<string, { text: string; ok: boolean }>>({});

  // Invites
  const [invites, setInvites] = useState<InviteData[]>([]);
  const [invitesLoading, setInvitesLoading] = useState(false);
  const [inviteMsg, setInviteMsg] = useState<{ text: string; ok: boolean } | null>(null);

  const myUserId = profile?.id ?? '';

  // ── Load data when logged in ──────────────────────────────────────────────

  const loadProfile = useCallback(async () => {
    try {
      const res = await fetch('/auth/me', { headers: authHeaders() });
      if (res.ok) setProfile(await res.json());
    } catch { /* ignore */ }
  }, []);

  const loadTeams = useCallback(async () => {
    setTeamsLoading(true);
    try {
      const res = await fetch('/teamadmin/teams', { headers: authHeaders() });
      if (res.ok) setTeams(await res.json());
    } finally {
      setTeamsLoading(false);
    }
  }, []);

  const loadInvites = useCallback(async () => {
    setInvitesLoading(true);
    try {
      const res = await fetch('/teamadmin/invites', { headers: authHeaders() });
      if (!res.ok) { setInvitesLoading(false); return; }
      const data: InviteData[] = await res.json();
      const enriched = await Promise.all(data.map(async inv => {
        try {
          const r = await fetch(`/teamadmin/${inv.sendingTeam}`, { headers: authHeaders() });
          if (r.ok) {
            const t = await r.json();
            return { ...inv, teamName: t.name };
          }
        } catch { /* ignore */ }
        return inv;
      }));
      setInvites(enriched);
    } finally {
      setInvitesLoading(false);
    }
  }, []);

  useEffect(() => {
    if (currentUser) {
      loadProfile();
      loadTeams();
      loadInvites();
    }
  }, [currentUser, loadProfile, loadTeams, loadInvites]);

  // ── Auth form ─────────────────────────────────────────────────────────────

  function switchAuthTab(t: AuthTab) {
    setAuthTab(t);
    setAuthError('');
    setUsername(''); setEmail(''); setPassword('');
  }

  async function handleAuthSubmit(e: React.FormEvent) {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);
    try {
      if (authTab === 'login') {
        const res = await fetch('/auth/login', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw { data, status: res.status };
        onSuccess(data.access_token, username);
      } else {
        const res = await fetch('/auth/register', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, email, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw { data, status: res.status };
        onSuccess(data.access_token, username);
      }
    } catch (err: unknown) {
      const e = err as { data?: Record<string, string>; status?: number };
      setAuthError(friendlyError(e.data ?? {}, e.status ?? 0, authTab));
    } finally {
      setAuthLoading(false);
    }
  }

  // ── Profile updates ───────────────────────────────────────────────────────

  async function handleProfileSave(e: React.FormEvent) {
    e.preventDefault();
    if (newPw && newPw !== confirmPw) {
      setProfileMsg({ text: 'New passwords do not match.', ok: false }); return;
    }
    setProfileLoading(true); setProfileMsg(null);
    try {
      const body: Record<string, string> = {};
      if (newUsername && newUsername !== profile?.username) body.username = newUsername;
      if (newEmail && newEmail !== profile?.email) body.email = newEmail;
      if (newPw) { body.currentPassword = currentPw; body.newPassword = newPw; }

      if (Object.keys(body).length === 0) {
        setProfileMsg({ text: 'No changes to save.', ok: false }); return;
      }

      const res = await fetch('/auth/profile', {
        method: 'PATCH', headers: authHeaders(), body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw { data, status: res.status };

      if (data.access_token) localStorage.setItem('c9_token', data.access_token);
      await loadProfile();
      setNewUsername(''); setNewEmail(''); setCurrentPw(''); setNewPw(''); setConfirmPw('');
      setProfileMsg({ text: 'Profile updated successfully.', ok: true });
    } catch (err: unknown) {
      const e = err as { data?: Record<string, string>; status?: number };
      setProfileMsg({ text: friendlyError(e.data ?? {}, e.status ?? 0, 'profile'), ok: false });
    } finally {
      setProfileLoading(false);
    }
  }

  async function handleDeleteAccount() {
    if (!window.confirm('Delete your account? This cannot be undone.')) return;
    await fetch('/teamadmin/account', { method: 'DELETE', headers: authHeaders() });
    onLogout();
  }

  // ── Teams ─────────────────────────────────────────────────────────────────

  async function toggleTeam(teamId: string) {
    if (expandedTeam === teamId) { setExpandedTeam(null); return; }
    setExpandedTeam(teamId);
    if (!teamMembers[teamId]) {
      const [mRes, kRes] = await Promise.all([
        fetch(`/teamadmin/${teamId}/members/rich`, { headers: authHeaders() }),
        fetch(`/teamadmin/${teamId}/apikeys`, { headers: authHeaders() }),
      ]);
      if (mRes.ok) {
        const members = await mRes.json();
        setTeamMembers(prev => ({ ...prev, [teamId]: members }));
      }
      if (kRes.ok) {
        const keys = await kRes.json();
        setTeamKeys(prev => ({ ...prev, [teamId]: keys }));
      }
    }
  }

  async function handleCreateTeam(e: React.FormEvent) {
    e.preventDefault();
    if (!newTeamName.trim()) {
      setNewTeamNameError('Team name cannot be empty.');
      return;
    }
    setNewTeamNameError('');
    setTeamMsg(null);
    const res = await fetch('/teamadmin/create', {
      method: 'POST', headers: authHeaders(), body: JSON.stringify({ name: newTeamName.trim() }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok) {
      setNewTeamName('');
      setTeamMsg({ text: `Team "${data.team_name ?? newTeamName}" created.`, ok: true });
      loadTeams();
    } else {
      const errMsg = data.message === 'team_name_taken'
        ? 'A team with that name already exists.'
        : 'Failed to create team.';
      setNewTeamNameError(errMsg);
    }
  }

  async function handleCreateKey(e: React.FormEvent, teamId: string) {
    e.preventDefault();
    const name = (newKeyName[teamId] ?? '').trim();
    if (!name) return;
    const res = await fetch(`/teamadmin/${teamId}/apikeys`, {
      method: 'POST', headers: authHeaders(), body: JSON.stringify({ name }),
    });
    if (!res.ok) return;
    const key: ApiKeyData = await res.json();
    setTeamKeys(prev => ({ ...prev, [teamId]: [key, ...(prev[teamId] ?? [])] }));
    setNewKeyName(prev => ({ ...prev, [teamId]: '' }));
    if (key.key) setRevealedKey(prev => ({ ...prev, [key.id]: key.key! }));
  }

  async function handleDeleteKey(teamId: string, keyId: string) {
    const res = await fetch(`/teamadmin/${teamId}/apikeys/${keyId}`, {
      method: 'DELETE', headers: authHeaders(),
    });
    if (res.ok) {
      setTeamKeys(prev => ({ ...prev, [teamId]: (prev[teamId] ?? []).filter(k => k.id !== keyId) }));
      setRevealedKey(prev => { const n = { ...prev }; delete n[keyId]; return n; });
    }
  }

  async function handleRemoveMember(teamId: string, memberId: string) {
    const res = await fetch(`/teamadmin/${teamId}/members/${memberId}`, {
      method: 'DELETE', headers: authHeaders(),
    });
    if (res.ok) {
      setTeamMembers(prev => ({
        ...prev, [teamId]: (prev[teamId] ?? []).filter(m => m.userId !== memberId),
      }));
    }
  }

  async function handleLeaveTeam(teamId: string) {
    const res = await fetch(`/teamadmin/${teamId}/members/me`, {
      method: 'DELETE', headers: authHeaders(),
    });
    if (res.ok) { setTeams(prev => prev.filter(t => t.id !== teamId)); setExpandedTeam(null); }
  }

  async function handleDeleteTeam(teamId: string, teamName: string) {
    if (!window.confirm(`Delete team "${teamName}"? This cannot be undone.`)) return;
    const res = await fetch(`/teamadmin/delete/${teamId}`, {
      method: 'DELETE', headers: authHeaders(),
    });
    if (res.ok) { setTeams(prev => prev.filter(t => t.id !== teamId)); setExpandedTeam(null); }
  }

  async function handlePromote(teamId: string, memberId: string, currentRole: string) {
    const newRole = currentRole === 'member' ? 'admin' : 'member';
    const res = await fetch(`/teamadmin/${teamId}/members/${memberId}/role`, {
      method: 'PATCH', headers: authHeaders(), body: JSON.stringify({ role: newRole }),
    });
    if (res.ok) {
      setTeamMembers(prev => ({
        ...prev,
        [teamId]: (prev[teamId] ?? []).map(m =>
          m.userId === memberId ? { ...m, role: newRole } : m,
        ),
      }));
    }
  }

  function copyKey(id: string, key: string) {
    navigator.clipboard.writeText(key);
    setCopiedKey(id);
    setTimeout(() => setCopiedKey(null), 2000);
  }

  function dismissKey(id: string) {
    setRevealedKey(prev => { const n = { ...prev }; delete n[id]; return n; });
  }

  async function handleSendInvite(teamId: string) {
    const query = (inviteQuery[teamId] ?? '').trim();
    if (!query) return;
    setInviteSearchMsg(prev => ({ ...prev, [teamId]: { text: '', ok: true } }));
    const res = await fetch(`/teamadmin/${teamId}/invite`, {
      method: 'POST', headers: authHeaders(),
      body: JSON.stringify({ usernameOrEmail: query }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok) {
      setInviteQuery(prev => ({ ...prev, [teamId]: '' }));
      setInviteSearchMsg(prev => ({ ...prev, [teamId]: { text: `Invite sent to "${query}".`, ok: true } }));
    } else {
      const msg = data.message === 'user_not_found' ? 'No user found with that username or email.'
        : data.message === 'already_member' ? 'That user is already in this team.'
        : data.message === 'already_invited' ? 'That user has already been invited.'
        : 'Failed to send invite.';
      setInviteSearchMsg(prev => ({ ...prev, [teamId]: { text: msg, ok: false } }));
    }
  }

  // ── Invites ───────────────────────────────────────────────────────────────

  async function handleAcceptInvite(inviteId: string) {
    const res = await fetch(`/teamadmin/invite/${inviteId}/accept`, {
      method: 'POST', headers: authHeaders(),
    });
    if (res.ok) {
      setInvites(prev => prev.filter(i => i.id !== inviteId));
      setInviteMsg({ text: 'Invite accepted! You are now a team member.', ok: true });
      loadTeams();
    } else {
      setInviteMsg({ text: 'Failed to accept invite.', ok: false });
    }
  }

  async function handleRejectInvite(inviteId: string) {
    const res = await fetch(`/teamadmin/invite/${inviteId}`, {
      method: 'DELETE', headers: authHeaders(),
    });
    if (res.ok) setInvites(prev => prev.filter(i => i.id !== inviteId));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  const inputClass =
    'w-full pl-9 pr-4 py-2.5 rounded-xl border-2 border-c9-cyan/40 focus:border-c9-cyan focus:outline-none bg-white/60 text-c9-text placeholder-c9-muted text-sm';
  const btnPrimary =
    'w-full py-2.5 bg-c9-cyan text-white font-bold rounded-xl hover:brightness-105 transition disabled:opacity-50';
  const btnOutline =
    'flex-1 py-2 rounded-xl border-2 border-c9-cyan text-c9-cyan font-semibold text-sm hover:bg-c9-cyan hover:text-white transition';

  return (
    <div className="min-h-screen flex items-start justify-center pt-10 pb-10 px-4">
      <div className="w-full max-w-2xl bg-white bg-opacity-85 backdrop-blur-md rounded-2xl border-2 border-c9-cyan shadow-xl">

        {/* ── Header ── */}
        <div className="flex items-center justify-between px-8 pt-7 pb-4 border-b border-c9-cyan/20">
          <h2 className="text-2xl font-bold tracking-wide">
            <span className="text-c9-cyan font-extrabold">C9</span>
            <span className="text-c9-text"> Account</span>
          </h2>
          <button onClick={onClose} className="text-c9-muted hover:text-c9-text transition" aria-label="Close">
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* ══════════════════════════════════════════
            NOT LOGGED IN — login / register forms
        ══════════════════════════════════════════ */}
        {!currentUser ? (
          <div className="px-8 py-6">
            {/* Tabs */}
            <div className="flex rounded-xl overflow-hidden border-2 border-c9-cyan mb-6">
              {(['login', 'register'] as AuthTab[]).map(t => (
                <button key={t} onClick={() => switchAuthTab(t)}
                  className={`flex-1 py-2 font-semibold text-sm transition ${authTab === t ? 'bg-c9-cyan text-white' : 'text-c9-cyan hover:bg-c9-cyan/10'}`}>
                  {t === 'login' ? 'Login' : 'Sign Up'}
                </button>
              ))}
            </div>

            <form onSubmit={handleAuthSubmit} className="space-y-4">
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)}
                  required autoComplete="username" className={inputClass} />
              </div>
              {authTab === 'register' && (
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                  <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)}
                    required autoComplete="email" className={inputClass} />
                </div>
              )}
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)}
                  required autoComplete={authTab === 'login' ? 'current-password' : 'new-password'} className={inputClass} />
              </div>
              {authError && <p className="text-sm text-red-500 text-center">{authError}</p>}
              <button type="submit" disabled={authLoading} className={btnPrimary}>
                {authLoading ? '...' : authTab === 'login' ? 'Login' : 'Create Account'}
              </button>
            </form>
          </div>

        ) : (
          /* ══════════════════════════════════════════
              LOGGED IN — tabbed account panel
          ══════════════════════════════════════════ */
          <>
            {/* Tab bar */}
            <div className="flex border-b border-c9-cyan/20">
              {[
                { id: 'profile', label: 'Profile', icon: <User className="w-4 h-4" /> },
                { id: 'teams',   label: 'My Teams', icon: <Users className="w-4 h-4" /> },
                { id: 'invites', label: `Invites${invites.length > 0 ? ` (${invites.length})` : ''}`, icon: <Bell className="w-4 h-4" /> },
              ].map(tab => (
                <button key={tab.id} onClick={() => setPanelTab(tab.id as PanelTab)}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-sm font-semibold transition border-b-2 ${
                    panelTab === tab.id
                      ? 'border-c9-cyan text-c9-cyan bg-c9-cyan/5'
                      : 'border-transparent text-c9-muted hover:text-c9-text'
                  }`}>
                  {tab.icon}{tab.label}
                </button>
              ))}
            </div>

            <div className="px-8 py-6 max-h-[75vh] overflow-y-auto">

              {/* ── Profile tab ── */}
              {panelTab === 'profile' && (
                <div className="space-y-6">
                  {/* Avatar */}
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-full bg-c9-cyan flex items-center justify-center border-2 border-white shadow">
                      <span className="text-white text-xl font-bold">
                        {(profile?.username ?? currentUser).slice(0, 2).toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <p className="text-c9-text font-bold text-lg">{profile?.username ?? currentUser}</p>
                      <p className="text-c9-muted text-sm">{profile?.email}</p>
                    </div>
                  </div>

                  <form onSubmit={handleProfileSave} className="space-y-4">
                    <p className="text-xs text-c9-muted uppercase tracking-widest font-semibold">Update Info</p>

                    <div className="relative">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                      <input type="text" placeholder={`Username (current: ${profile?.username ?? ''})`}
                        value={newUsername} onChange={e => setNewUsername(e.target.value)}
                        autoComplete="username" className={inputClass} />
                    </div>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                      <input type="email" placeholder={`Email (current: ${profile?.email ?? ''})`}
                        value={newEmail} onChange={e => setNewEmail(e.target.value)}
                        autoComplete="email" className={inputClass} />
                    </div>

                    <p className="text-xs text-c9-muted uppercase tracking-widest font-semibold pt-2">Change Password</p>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                      <input type="password" placeholder="Current password" value={currentPw}
                        onChange={e => setCurrentPw(e.target.value)} autoComplete="current-password" className={inputClass} />
                    </div>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                      <input type="password" placeholder="New password" value={newPw}
                        onChange={e => setNewPw(e.target.value)} autoComplete="new-password" className={inputClass} />
                    </div>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-c9-muted" />
                      <input type="password" placeholder="Confirm new password" value={confirmPw}
                        onChange={e => setConfirmPw(e.target.value)} autoComplete="new-password" className={inputClass} />
                    </div>

                    {profileMsg && (
                      <p className={`text-sm text-center font-medium ${profileMsg.ok ? 'text-emerald-600' : 'text-red-500'}`}>
                        {profileMsg.text}
                      </p>
                    )}

                    <button type="submit" disabled={profileLoading} className={btnPrimary}>
                      {profileLoading ? '...' : 'Save Changes'}
                    </button>
                  </form>

                  <div className="flex gap-3 pt-2">
                    <button onClick={onLogout} className={btnOutline}>
                      <LogOut className="w-4 h-4 inline mr-1" />Sign Out
                    </button>
                    <button onClick={handleDeleteAccount}
                      className="flex-1 py-2 rounded-xl border-2 border-red-400 text-red-500 font-semibold text-sm hover:bg-red-50 transition">
                      <Trash2 className="w-4 h-4 inline mr-1" />Delete Account
                    </button>
                  </div>
                </div>
              )}

              {/* ── Teams tab ── */}
              {panelTab === 'teams' && (
                <div className="space-y-4">
                  {/* Create team */}
                  <form onSubmit={handleCreateTeam} className="space-y-1">
                    <div className="flex gap-2">
                      <input type="text" placeholder="New team name…" value={newTeamName}
                        onChange={e => { setNewTeamName(e.target.value); if (newTeamNameError) setNewTeamNameError(''); }}
                        className={`flex-1 px-3 py-2 rounded-xl border-2 focus:outline-none text-sm bg-white/60 text-c9-text placeholder-c9-muted transition ${
                          newTeamNameError ? 'border-red-400 focus:border-red-400' : 'border-c9-cyan/40 focus:border-c9-cyan'
                        }`} />
                      <button type="submit"
                        className="px-4 py-2 bg-c9-cyan text-white rounded-xl font-semibold text-sm hover:brightness-105 transition flex items-center gap-1">
                        <Plus className="w-4 h-4" />Create
                      </button>
                    </div>
                    {newTeamNameError && (
                      <p className="text-xs text-red-500 pl-1">{newTeamNameError}</p>
                    )}
                  </form>
                  {teamMsg && (
                    <p className={`text-sm text-center ${teamMsg.ok ? 'text-emerald-600' : 'text-red-500'}`}>{teamMsg.text}</p>
                  )}

                  {teamsLoading && <p className="text-c9-muted text-sm text-center">Loading…</p>}

                  {!teamsLoading && teams.length === 0 && (
                    <p className="text-c9-muted text-sm text-center py-4">You are not in any teams yet.</p>
                  )}

                  {teams.map(team => {
                    const isOwner = team.ownerUserId === myUserId;
                    const members = teamMembers[team.id] ?? [];
                    const myMembership = members.find(m => m.userId === myUserId);
                    const isAdmin = isOwner || myMembership?.role === 'admin';
                    const expanded = expandedTeam === team.id;

                    return (
                      <div key={team.id} className="rounded-2xl border-2 border-c9-cyan/30 overflow-hidden">
                        {/* Team header */}
                        <button onClick={() => toggleTeam(team.id)}
                          className="w-full flex items-center justify-between px-4 py-3 bg-white/60 hover:bg-c9-cyan/5 transition">
                          <div className="flex items-center gap-2">
                            <Users className="w-4 h-4 text-c9-cyan" />
                            <span className="font-semibold text-c9-text">{team.name}</span>
                            <RoleBadge role={isOwner ? 'owner' : myMembership?.role ?? 'member'} />
                          </div>
                          {expanded ? <ChevronDown className="w-4 h-4 text-c9-muted" /> : <ChevronRight className="w-4 h-4 text-c9-muted" />}
                        </button>

                        {expanded && (
                          <div className="px-4 pb-4 pt-2 space-y-4 bg-white/30">

                            {/* Members */}
                            <div>
                              <p className="text-xs text-c9-muted uppercase tracking-widest font-semibold mb-2">Members</p>
                              {members.length === 0
                                ? <p className="text-c9-muted text-xs">No members yet.</p>
                                : members.map(m => (
                                  <div key={m.userId} className="flex items-center justify-between py-1.5 border-b border-c9-cyan/10 last:border-0">
                                    <div className="flex items-center gap-2">
                                      <div className="w-7 h-7 rounded-full bg-c9-cyan/20 flex items-center justify-center text-xs font-bold text-c9-cyan">
                                        {m.username.slice(0, 2).toUpperCase()}
                                      </div>
                                      <span className="text-sm text-c9-text font-medium">{m.username}</span>
                                      <RoleBadge role={m.userId === team.ownerUserId ? 'owner' : m.role} />
                                    </div>
                                    {isAdmin && m.userId !== myUserId && m.userId !== team.ownerUserId && (isOwner || m.role === 'member') && (
                                      <div className="flex items-center gap-1">
                                        <button onClick={() => handlePromote(team.id, m.userId, m.role)}
                                          title={m.role === 'member' ? 'Promote to admin' : 'Demote to member'}
                                          className="p-1 rounded hover:bg-c9-cyan/10 text-c9-muted hover:text-c9-cyan transition">
                                          <RefreshCw className="w-3.5 h-3.5" />
                                        </button>
                                        <button onClick={() => handleRemoveMember(team.id, m.userId)}
                                          title="Remove member"
                                          className="p-1 rounded hover:bg-red-50 text-c9-muted hover:text-red-500 transition">
                                          <XCircle className="w-3.5 h-3.5" />
                                        </button>
                                      </div>
                                    )}
                                  </div>
                                ))
                              }
                            </div>

                            {/* API Keys (admin/owner only) */}
                            {isAdmin && (
                              <div>
                                <p className="text-xs text-c9-muted uppercase tracking-widest font-semibold mb-2">API Keys</p>
                                <form onSubmit={e => handleCreateKey(e, team.id)} className="flex gap-2 mb-2">
                                  <input type="text" placeholder="Key name…"
                                    value={newKeyName[team.id] ?? ''}
                                    onChange={e => setNewKeyName(prev => ({ ...prev, [team.id]: e.target.value }))}
                                    className="flex-1 px-3 py-1.5 rounded-lg border border-c9-cyan/40 focus:border-c9-cyan focus:outline-none text-xs bg-white/60" />
                                  <button type="submit"
                                    className="px-3 py-1.5 bg-c9-cyan text-white rounded-lg text-xs font-semibold hover:brightness-105 transition flex items-center gap-1">
                                    <Key className="w-3 h-3" />Generate
                                  </button>
                                </form>

                                {(teamKeys[team.id] ?? []).length === 0
                                  ? <p className="text-c9-muted text-xs">No API keys.</p>
                                  : (teamKeys[team.id] ?? []).map(k => (
                                    <div key={k.id} className="py-1.5 border-b border-c9-cyan/10 last:border-0">
                                      {revealedKey[k.id] ? (
                                        /* One-time reveal banner */
                                        <div className="rounded-lg border border-amber-300 bg-amber-50 p-2 space-y-1">
                                          <div className="flex items-center gap-1 text-amber-700 text-xs font-semibold">
                                            <AlertTriangle className="w-3 h-3" />
                                            Copy this key now — it won't be shown again.
                                          </div>
                                          <div className="flex items-center gap-1">
                                            <code className="flex-1 text-xs text-emerald-700 bg-white border border-emerald-200 px-2 py-1 rounded truncate">
                                              {revealedKey[k.id]}
                                            </code>
                                            <button onClick={() => copyKey(k.id, revealedKey[k.id])}
                                              title="Copy key"
                                              className="p-1 rounded hover:bg-emerald-100 text-c9-muted hover:text-emerald-700 transition">
                                              {copiedKey === k.id ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                                            </button>
                                            <button onClick={() => dismissKey(k.id)}
                                              title="Dismiss (key will not be shown again)"
                                              className="p-1 rounded hover:bg-red-50 text-c9-muted hover:text-red-500 transition">
                                              <X className="w-3.5 h-3.5" />
                                            </button>
                                          </div>
                                          <p className="text-xs text-c9-muted">{k.name}</p>
                                        </div>
                                      ) : (
                                        <div className="flex items-center justify-between">
                                          <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-c9-text truncate">{k.name}</p>
                                            <p className="text-xs text-c9-muted">Prefix: {k.keyPrefix}…</p>
                                          </div>
                                          <button onClick={() => handleDeleteKey(team.id, k.id)}
                                            className="p-1 ml-2 rounded hover:bg-red-50 text-c9-muted hover:text-red-500 transition">
                                            <Trash2 className="w-3.5 h-3.5" />
                                          </button>
                                        </div>
                                      )}
                                    </div>
                                  ))
                                }
                              </div>
                            )}

                            {/* Invite members (admin/owner) */}
                            {isAdmin && (
                              <div>
                                <p className="text-xs text-c9-muted uppercase tracking-widest font-semibold mb-2">Invite Member</p>
                                <div className="flex gap-2">
                                  <div className="relative flex-1">
                                    <UserPlus className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-c9-muted" />
                                    <input
                                      type="text"
                                      placeholder="Username or email…"
                                      value={inviteQuery[team.id] ?? ''}
                                      onChange={e => {
                                        setInviteQuery(prev => ({ ...prev, [team.id]: e.target.value }));
                                        if (inviteSearchMsg[team.id]) setInviteSearchMsg(prev => ({ ...prev, [team.id]: { text: '', ok: true } }));
                                      }}
                                      onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); handleSendInvite(team.id); } }}
                                      className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-c9-cyan/40 focus:border-c9-cyan focus:outline-none text-xs bg-white/60" />
                                  </div>
                                  <button
                                    onClick={() => handleSendInvite(team.id)}
                                    className="px-3 py-1.5 bg-c9-cyan text-white rounded-lg text-xs font-semibold hover:brightness-105 transition flex items-center gap-1">
                                    <UserPlus className="w-3 h-3" />Invite
                                  </button>
                                </div>
                                {inviteSearchMsg[team.id]?.text && (
                                  <p className={`text-xs mt-1 pl-1 ${inviteSearchMsg[team.id].ok ? 'text-emerald-600' : 'text-red-500'}`}>
                                    {inviteSearchMsg[team.id].text}
                                  </p>
                                )}
                              </div>
                            )}

                            {/* Leave / Delete */}
                            <div className="flex gap-2 pt-1">
                              {!isOwner && (
                                <button onClick={() => handleLeaveTeam(team.id)}
                                  className="flex-1 py-1.5 text-xs rounded-lg border border-c9-muted/50 text-c9-muted hover:bg-gray-50 transition font-semibold">
                                  Leave Team
                                </button>
                              )}
                              {isOwner && (
                                <button onClick={() => handleDeleteTeam(team.id, team.name)}
                                  className="flex-1 py-1.5 text-xs rounded-lg border border-red-300 text-red-500 hover:bg-red-50 transition font-semibold">
                                  Delete Team
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* ── Invites tab ── */}
              {panelTab === 'invites' && (
                <div className="space-y-3">
                  {inviteMsg && (
                    <p className={`text-sm text-center font-medium ${inviteMsg.ok ? 'text-emerald-600' : 'text-red-500'}`}>
                      {inviteMsg.text}
                    </p>
                  )}
                  {invitesLoading && <p className="text-c9-muted text-sm text-center">Loading…</p>}
                  {!invitesLoading && invites.length === 0 && (
                    <div className="text-center py-8">
                      <Bell className="w-10 h-10 text-c9-cyan/30 mx-auto mb-2" />
                      <p className="text-c9-muted text-sm">No pending invites.</p>
                    </div>
                  )}
                  {invites.map(inv => (
                    <div key={inv.id} className="flex items-center justify-between p-4 rounded-2xl border-2 border-c9-cyan/30 bg-white/50">
                      <div>
                        <p className="font-semibold text-c9-text">
                          {inv.teamName ?? `Team ${inv.sendingTeam.slice(0, 8)}…`}
                        </p>
                        <p className="text-xs text-c9-muted mt-0.5">
                          {new Date(inv.createdAt).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={() => handleAcceptInvite(inv.id)}
                          className="flex items-center gap-1 px-3 py-1.5 bg-c9-cyan text-white rounded-lg text-sm font-semibold hover:brightness-105 transition">
                          <Check className="w-3.5 h-3.5" />Accept
                        </button>
                        <button onClick={() => handleRejectInvite(inv.id)}
                          className="flex items-center gap-1 px-3 py-1.5 border-2 border-c9-muted/40 text-c9-muted rounded-lg text-sm font-semibold hover:bg-gray-50 transition">
                          <X className="w-3.5 h-3.5" />Decline
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

            </div>
          </>
        )}
      </div>
    </div>
  );
}
