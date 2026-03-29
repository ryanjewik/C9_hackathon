-- Shared schema for initial DB setup
-- Idempotent: safe to run multiple times before/after restores

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =========================
-- USERS
-- =========================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- TEAMS
-- =========================
CREATE TABLE IF NOT EXISTS teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner_user_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_team_owner
        FOREIGN KEY (owner_user_id)
        REFERENCES users(id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS invitations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sending_team UUID NOT NULL,
    receiving_player UUID NOT NULL,
    sending_admin UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_sending_team
        FOREIGN KEY (sending_team)
        REFERENCES teams(id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_receiving_player
        FOREIGN KEY (receiving_player)
        REFERENCES users(id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_sending_admin
        FOREIGN KEY (sending_admin)
        REFERENCES users(id)
        ON DELETE CASCADE
);

-- =========================
-- TEAM MEMBERS (JOIN TABLE)
-- =========================
CREATE TABLE IF NOT EXISTS team_members (
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
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active',
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
CREATE TABLE IF NOT EXISTS vods (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    team_id UUID NOT NULL,
    video_link TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded',
    result_path TEXT,
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
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- API key lookup (critical for /auth/token)
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

-- Team queries
CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id);
CREATE INDEX IF NOT EXISTS idx_vods_team ON vods(team_id);

-- Optional: recent activity
CREATE INDEX IF NOT EXISTS idx_api_keys_last_used ON api_keys(last_used_at);