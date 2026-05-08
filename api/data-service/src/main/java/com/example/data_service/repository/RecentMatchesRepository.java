package com.example.data_service.repository;

import com.example.data_service.entity.Match;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface RecentMatchesRepository extends JpaRepository<Match, Integer> {
    @Query(value = "SELECT m.* FROM esports_matches AS m INNER JOIN esports_tournaments AS t ON t.id = m.tournament_id WHERE t.tier = 'VCT' ORDER BY m.date DESC LIMIT 5",
    nativeQuery = true)
    List<Match> findRecentMatches();

    @Query(value = """
        SELECT
            m.date,
            CASE WHEN m.winner = :teamId THEN true ELSE false END AS won,
            CASE WHEN m.team_1_id = :teamId THEN m.team_2_name ELSE m.team_1_name END AS opponent_name,
            CASE WHEN m.team_1_id = :teamId THEN m.team_1_score ELSE m.team_2_score END AS team_score,
            CASE WHEN m.team_1_id = :teamId THEN m.team_2_score ELSE m.team_1_score END AS opponent_score,
            m.tournament_name
        FROM esports_matches m
        INNER JOIN esports_tournaments t ON t.id = m.tournament_id
        WHERE (m.team_1_id = :teamId OR m.team_2_id = :teamId)
          AND t.tier = 'VCT'
        ORDER BY m.date DESC
        LIMIT 15
        """, nativeQuery = true)
    List<Object[]> findTeamMatchHistory(@Param("teamId") Integer teamId);
}
