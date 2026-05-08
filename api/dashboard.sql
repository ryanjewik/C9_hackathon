SELECT p.nickname, pg.agent, AVG(pg.rating) as average_rating, SUM(pg.kills) AS kills, SUM(pg.deaths) AS deaths, SUM(pg.assists) AS assists, SUM(pg.fk) AS first_kills, SUM(pg.fd) AS first_deaths
FROM esports_matches AS m
INNER JOIN esports_tournaments as t ON m.tournament_id = t.id
INNER JOIN esports_player_games as pg ON pg.match_id = m.id
INNER JOIN esports_players as p ON p.id = pg.player_id
INNER JOIN esports_teams as tm ON pg.team_id = tm.id
WHERE m.date >= CURRENT_DATE - INTERVAL '1 week' AND t.tier = 'VCT'
GROUP BY p.nickname, pg.agent
ORDER BY p.nickname

SELECT p.nickname, ARRAY_AGG(DISTINCT pg.agent), AVG(pg.rating) as average_rating, SUM(pg.kills) AS kills, SUM(pg.deaths) AS deaths, SUM(pg.assists) AS assists, SUM(pg.fk) AS first_kills, SUM(pg.fd) AS first_deaths
FROM esports_matches AS m
INNER JOIN esports_tournaments as t ON m.tournament_id = t.id
INNER JOIN esports_player_games as pg ON pg.match_id = m.id
INNER JOIN esports_players as p ON p.id = pg.player_id
INNER JOIN esports_teams as tm ON pg.team_id = tm.id
WHERE m.date >= CURRENT_DATE - INTERVAL '2 weeks' AND t.tier = 'VCT'
GROUP BY p.nickname
ORDER BY average_rating DESC
LIMIT 10


SELECT * FROM esports_tournaments WHERE status = 'ongoing' AND tier = 'VCT'

SELECT * FROM esports_matches INNER JOIN esports_tournaments ON esports_tournaments.id = esports_matches.tournament_id WHERE tier = 'VCT' ORDER BY date DESC LIMIT 3

WITH ban_counts AS (SELECT mv.map_selected AS map, COUNT(map_selected) AS ban_count
	FROM esports_map_veto AS mv 
	INNER JOIN esports_matches AS m ON m.id = mv.match_id 
	WHERE date >= CURRENT_DATE - INTERVAL '3 months' AND type = 'ban' AND team_id = 2059
	GROUP BY mv.map_selected
)
SELECT mv.map_selected, COUNT(map_selected) - b.ban_count AS pick_count, b.ban_count
FROM esports_map_veto AS mv 
INNER JOIN esports_matches AS m ON m.id = mv.match_id
INNER JOIN ban_counts AS b ON b.map = mv.map_selected
WHERE date >= CURRENT_DATE - INTERVAL '3 months' AND team_id = 2059
GROUP BY mv.map_selected, ban_count

WITH total_matches AS (SELECT pg.tournament_id, t.name, COUNT(DISTINCT game_id) AS total_matches FROM esports_player_games AS pg 
	INNER JOIN esports_tournaments AS t ON t.id = pg.tournament_id 
	WHERE EXTRACT(YEAR FROM end_date) = 2026 AND t.tier = 'VCT' GROUP BY pg.tournament_id, t.name
)
SELECT t.name, agent, COUNT(*) AS agent_picks, tm.total_matches, COUNT(*)::DECIMAL / (tm.total_matches * 2)::DECIMAL AS pickrates
FROM esports_player_games AS pg
INNER JOIN esports_tournaments AS t ON t.id = pg.tournament_id 
INNER JOIN total_matches as tm ON tm.tournament_id = pg.tournament_id
WHERE EXTRACT(YEAR FROM end_date) = 2026 AND t.tier = 'VCT' 
GROUP BY t.name, agent, tm.total_matches

SELECT t.name, g.map, COUNT(map) 
FROM esports_game_scores as g 
INNER JOIN esports_matches AS m ON g.match_id = m.id
INNER JOIN esports_tournaments as t ON m.tournament_id = t.id
WHERE t.tier = 'VCT' AND EXTRACT(YEAR FROM t.end_date) = 2026
GROUP BY t.name, g.map;