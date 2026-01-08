-- VLR.gg Esports Database Schema
-- Database: cloud9

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS esports_player_games CASCADE;
DROP TABLE IF EXISTS esports_game_scores CASCADE;
DROP TABLE IF EXISTS esports_rosters CASCADE;
DROP TABLE IF EXISTS esports_map_veto CASCADE;
DROP TABLE IF EXISTS esports_matches CASCADE;
DROP TABLE IF EXISTS esports_players CASCADE;
DROP TABLE IF EXISTS esports_tournament_placements CASCADE;
DROP TABLE IF EXISTS esports_teams CASCADE;
DROP TABLE IF EXISTS esports_tournaments CASCADE;

-- Tournaments table
CREATE TABLE esports_tournaments (
    id INTEGER PRIMARY KEY,  -- VLR event ID from URL
    name VARCHAR(255) NOT NULL,
    start_date DATE,
    end_date DATE,
    prize_pool VARCHAR(100),
    location VARCHAR(255),
    status VARCHAR(20) CHECK (status IN ('upcoming', 'ongoing', 'completed'))
);

-- Teams table
CREATE TABLE esports_teams (
    id INTEGER PRIMARY KEY,  -- VLR team ID from URL
    name VARCHAR(255) NOT NULL,
    team_tag VARCHAR(10),
    location VARCHAR(100),
    titles INTEGER[] DEFAULT '{}',  -- Array of tournament IDs won
    match_wins INTEGER DEFAULT 0,
    match_losses INTEGER DEFAULT 0,
    current_roster_id INTEGER  -- Will be updated after rosters are created
);

-- Tournament placements
CREATE TABLE esports_tournament_placements (
    id SERIAL PRIMARY KEY,
    tournament_id INTEGER REFERENCES esports_tournaments(id),
    placement VARCHAR(20),  -- e.g., "1st", "5th-6th", "7th-8th"
    esports_team_id INTEGER REFERENCES esports_teams(id),
    prize_money VARCHAR(50),  -- e.g., "$1,000,000"
    stage VARCHAR(100),  -- e.g., "playoffs", "group-stage"
    players INTEGER[] DEFAULT '{}',  -- Array of player IDs who played for this team in the tournament
    UNIQUE(tournament_id, esports_team_id, stage)
);

-- Players table
CREATE TABLE esports_players (
    id INTEGER PRIMARY KEY,  -- VLR player ID from URL
    nickname VARCHAR(100) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    country VARCHAR(100),
    team_id INTEGER REFERENCES esports_teams(id),
    titles INTEGER[] DEFAULT '{}',  -- Array of tournament IDs won
    
    -- All-time stats
    all_time_maps INTEGER DEFAULT 0,
    all_time_map_wins INTEGER DEFAULT 0,
    all_time_map_losses INTEGER DEFAULT 0,
    all_time_rating DECIMAL(4, 2),
    all_time_acs DECIMAL(6, 2),
    all_time_kills INTEGER DEFAULT 0,
    all_time_deaths INTEGER DEFAULT 0,
    all_time_assists INTEGER DEFAULT 0,
    all_time_avg_kills DECIMAL(5, 2),
    all_time_avg_deaths DECIMAL(5, 2),
    all_time_avg_assists DECIMAL(5, 2),
    all_time_kast DECIMAL(5, 2),
    all_time_adr DECIMAL(6, 2),
    all_time_hs_percent DECIMAL(5, 2),
    all_time_fk INTEGER DEFAULT 0,
    all_time_fd INTEGER DEFAULT 0,
    all_time_avg_fk DECIMAL(5, 2),
    all_time_avg_fd DECIMAL(5, 2),
    
    -- Last 60 days stats
    last_60_maps INTEGER,
    last_60_map_wins INTEGER,
    last_60_map_losses INTEGER,
    last_60_rating DECIMAL(4, 2),
    last_60_acs DECIMAL(6, 2),
    last_60_kills INTEGER,
    last_60_deaths INTEGER,
    last_60_assists INTEGER,
    last_60_avg_kills DECIMAL(5, 2),
    last_60_avg_deaths DECIMAL(5, 2),
    last_60_avg_assists DECIMAL(5, 2),
    last_60_kast DECIMAL(5, 2),
    last_60_adr DECIMAL(6, 2),
    last_60_hs_percent DECIMAL(5, 2),
    last_60_fk INTEGER,
    last_60_fd INTEGER,
    last_60_avg_fk DECIMAL(5, 2),
    last_60_avg_fd DECIMAL(5, 2)
);

-- Matches table
CREATE TABLE esports_matches (
    id INTEGER PRIMARY KEY,  -- VLR match ID from URL
    phase VARCHAR(100),  -- e.g., "Group Stage: Opening (A)"
    date TIMESTAMP,
    patch VARCHAR(20),
    tournament_id INTEGER REFERENCES esports_tournaments(id),
    tournament_name VARCHAR(255),
    team_1_name VARCHAR(255),
    team_1_id INTEGER REFERENCES esports_teams(id),
    team_1_score INTEGER,
    team_2_name VARCHAR(255),
    team_2_id INTEGER REFERENCES esports_teams(id),
    team_2_score INTEGER,
    winner INTEGER REFERENCES esports_teams(id),
    format VARCHAR(10),  -- bo1, bo3, bo5
    map_1 VARCHAR(50),
    map_2 VARCHAR(50),
    map_3 VARCHAR(50),
    map_4 VARCHAR(50),
    map_5 VARCHAR(50)
);

-- Map veto table
CREATE TABLE esports_map_veto (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES esports_matches(id),
    type VARCHAR(10) CHECK (type IN ('ban', 'pick')),
    team_id INTEGER REFERENCES esports_teams(id),
    map_selected VARCHAR(50),
    turn INTEGER
);

-- Rosters table
CREATE TABLE esports_rosters (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES esports_teams(id),
    player_1 INTEGER REFERENCES esports_players(id),
    player_2 INTEGER REFERENCES esports_players(id),
    player_3 INTEGER REFERENCES esports_players(id),
    player_4 INTEGER REFERENCES esports_players(id),
    player_5 INTEGER REFERENCES esports_players(id),
    date_created DATE,
    map_wins INTEGER DEFAULT 0,
    map_losses INTEGER DEFAULT 0,
    UNIQUE(team_id, player_1, player_2, player_3, player_4, player_5)
);

-- Game scores table (individual maps)
CREATE TABLE esports_game_scores (
    id INTEGER PRIMARY KEY,  -- Game ID from URL param
    match_id INTEGER REFERENCES esports_matches(id),
    team_1_score INTEGER,
    team_2_score INTEGER,
    team_1_id INTEGER REFERENCES esports_teams(id),
    team_2_id INTEGER REFERENCES esports_teams(id),
    team_1_name VARCHAR(255),
    team_2_name VARCHAR(255),
    map VARCHAR(50),
    winner INTEGER REFERENCES esports_teams(id)
);

-- Player game stats table
CREATE TABLE esports_player_games (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES esports_matches(id),
    game_id INTEGER REFERENCES esports_game_scores(id),
    player_id INTEGER REFERENCES esports_players(id),
    team_id INTEGER REFERENCES esports_teams(id),
    roster_id INTEGER REFERENCES esports_rosters(id),
    tournament_id INTEGER REFERENCES esports_tournaments(id),
    map VARCHAR(50),
    agent VARCHAR(50),
    rating DECIMAL(4, 2),
    acs INTEGER,
    kills INTEGER,
    deaths INTEGER,
    assists INTEGER,
    kast VARCHAR(10),
    adr INTEGER,
    hs_percent VARCHAR(10),
    fk INTEGER,  -- First Kills
    fd INTEGER,  -- First Deaths
    opponent_roster_id INTEGER REFERENCES esports_rosters(id),
    opponent_team_id INTEGER REFERENCES esports_teams(id)
);

-- Create indexes for common queries
CREATE INDEX idx_matches_tournament ON esports_matches(tournament_id);
CREATE INDEX idx_matches_team1 ON esports_matches(team_1_id);
CREATE INDEX idx_matches_team2 ON esports_matches(team_2_id);
CREATE INDEX idx_player_games_player ON esports_player_games(player_id);
CREATE INDEX idx_player_games_match ON esports_player_games(match_id);
CREATE INDEX idx_player_games_game ON esports_player_games(game_id);
CREATE INDEX idx_placements_tournament ON esports_tournament_placements(tournament_id);
CREATE INDEX idx_rosters_team ON esports_rosters(team_id);
CREATE INDEX idx_map_veto_match ON esports_map_veto(match_id);
