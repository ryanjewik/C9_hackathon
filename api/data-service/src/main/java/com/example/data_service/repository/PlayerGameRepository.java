package com.example.data_service.repository;

import com.example.data_service.dto.PlayerGameDto;
import com.example.data_service.entity.PlayerGame;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface PlayerGameRepository extends JpaRepository<PlayerGame, Integer> {
    @Query("select new com.example.data_service.dto.PlayerGameDto(p.id, p.matchId, p.gameId, p.playerId, p.teamId, p.rosterId, p.tournamentId, p.map, p.agent, p.rating, p.acs, p.kills, p.deaths, p.assists, p.kast, p.adr, p.hsPercent, p.fk, p.fd) from PlayerGame p")
    Page<PlayerGameDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.PlayerGameDto(p.id, p.matchId, p.gameId, p.playerId, p.teamId, p.rosterId, p.tournamentId, p.map, p.agent, p.rating, p.acs, p.kills, p.deaths, p.assists, p.kast, p.adr, p.hsPercent, p.fk, p.fd) from PlayerGame p where p.id = :id")
    PlayerGameDto findDtoById(@Param("id") Integer id);

    @Query("SELECT p FROM PlayerGame p WHERE p.gameId IN :gameIds")
    java.util.List<com.example.data_service.entity.PlayerGame> findAllByGameIdIn(@Param("gameIds") java.util.List<Integer> gameIds);

    @Query(value = """
            SELECT p.nickname,
                   ARRAY_AGG(DISTINCT pg.agent) AS agents,
                   AVG(pg.rating) AS average_rating,
                   SUM(pg.kills) AS kills,
                   SUM(pg.deaths) AS deaths,
                   SUM(pg.assists) AS assists,
                   SUM(pg.fk) AS first_kills,
                   SUM(pg.fd) AS first_deaths
            FROM esports_matches AS m
            INNER JOIN esports_tournaments AS t ON m.tournament_id = t.id
            INNER JOIN esports_player_games AS pg ON pg.match_id = m.id
            INNER JOIN esports_players AS p ON p.id = pg.player_id
            INNER JOIN esports_teams AS tm ON pg.team_id = tm.id
            WHERE m.date >= CURRENT_DATE - INTERVAL '2 weeks' AND t.tier = 'VCT'
            GROUP BY p.nickname
            ORDER BY average_rating DESC
            LIMIT 10
            """, nativeQuery = true)
    java.util.List<Object[]> findTopPlayerStats();

    @Query(value = """
            WITH total_matches AS (
                SELECT pg.tournament_id, t.name, COUNT(DISTINCT pg.game_id) AS total_matches
                FROM esports_player_games AS pg
                INNER JOIN esports_tournaments AS t ON t.id = pg.tournament_id
                WHERE EXTRACT(YEAR FROM t.end_date) = 2026 AND t.tier = 'VCT'
                GROUP BY pg.tournament_id, t.name
            )
            SELECT t.name, pg.agent, COUNT(*) AS agent_picks, tm.total_matches,
                   COUNT(*)::DECIMAL / (tm.total_matches * 2)::DECIMAL AS pickrates
            FROM esports_player_games AS pg
            INNER JOIN esports_tournaments AS t ON t.id = pg.tournament_id
            INNER JOIN total_matches AS tm ON tm.tournament_id = pg.tournament_id
            WHERE EXTRACT(YEAR FROM t.end_date) = 2026 AND t.tier = 'VCT'
            GROUP BY t.name, pg.agent, tm.total_matches
            ORDER BY t.name, pickrates DESC
            """, nativeQuery = true)
    java.util.List<Object[]> findAgentPickRates();

    @Query(value = """
            WITH AgentStats AS (
                SELECT
                    p.nickname,
                    pg.agent,
                    AVG(pg.rating) AS average_rating,
                    SUM(pg.kills) AS kills,
                    SUM(pg.deaths) AS deaths,
                    SUM(pg.assists) AS assists,
                    SUM(pg.fk) AS first_kills,
                    SUM(pg.fd) AS first_deaths
                FROM esports_matches AS m
                INNER JOIN esports_tournaments AS t ON m.tournament_id = t.id
                INNER JOIN esports_player_games AS pg ON pg.match_id = m.id
                INNER JOIN esports_players AS p ON p.id = pg.player_id
                INNER JOIN esports_teams AS tm ON pg.team_id = tm.id
                WHERE m.date >= CURRENT_DATE - INTERVAL '90 days'
                  AND t.tier = 'VCT'
                  AND pg.rating IS NOT NULL
                GROUP BY p.nickname, pg.agent
            )
            SELECT DISTINCT ON (agent) *
            FROM AgentStats
            ORDER BY agent, average_rating DESC
            """, nativeQuery = true)
    java.util.List<Object[]> findTopPlayerPerAgent();
}
