-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS 'pgcrypto';

-- =========================
-- USERS
-- =========================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- TEAMS
-- =========================
CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner_user_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_team_owner
        FOREIGN KEY (owner_user_id)
        REFERENCES users(id)
        ON DELETE RESTRICT
);

-- =========================
-- TEAM MEMBERS (JOIN TABLE)
-- =========================
CREATE TABLE team_members (
    team_id UUID NOT NULL,
    user_id UUID NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (team_id, user_id),

    CONSTRAINT fk_team_members_team
        FOREIGN KEY (team_id)
        REFERENCES teams(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_team_members_user
        FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

-- =========================
-- API KEYS (TEAM LEVEL)
-- =========================
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL, -- for display (e.g. sk_live_abcd)
    key_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active', -- active, revoked
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,

    CONSTRAINT fk_api_keys_team
        FOREIGN KEY (team_id)
        REFERENCES teams(id)
        ON DELETE CASCADE
);

-- =========================
-- VODS
-- =========================
CREATE TABLE vods (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    team_id UUID NOT NULL,
    video_link TEXT NOT NULL, -- S3 path or similar
    status TEXT NOT NULL DEFAULT 'uploaded', -- uploaded, processing, complete, failed
    result_path TEXT, -- optional processed output
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_vods_team
        FOREIGN KEY (team_id)
        REFERENCES teams(id)
        ON DELETE CASCADE
);

-- =========================
-- INDEXES (IMPORTANT)
-- =========================

-- Fast lookup for login
CREATE INDEX idx_users_email ON users(email);

-- API key lookup (critical for /auth/token)
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- Team queries
CREATE INDEX idx_team_members_user ON team_members(user_id);
CREATE INDEX idx_vods_team ON vods(team_id);

-- Optional: recent activity
CREATE INDEX idx_api_keys_last_used ON api_keys(last_used_at);