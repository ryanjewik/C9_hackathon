package com.example.data_service.repository;

import com.example.data_service.dto.GameScoreDto;
import com.example.data_service.entity.GameScore;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface GameScoreRepository extends JpaRepository<GameScore, Integer> {
    @Query("select new com.example.data_service.dto.GameScoreDto(g.id, g.matchId, g.team1Score, g.team2Score, g.team1Id, g.team2Id, g.team1Name, g.team2Name, g.map, g.winner) from GameScore g")
    Page<GameScoreDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.GameScoreDto(g.id, g.matchId, g.team1Score, g.team2Score, g.team1Id, g.team2Id, g.team1Name, g.team2Name, g.map, g.winner) from GameScore g where g.id = :id")
    GameScoreDto findDtoById(@Param("id") Integer id);

    @Query("SELECT g FROM GameScore g WHERE g.matchId IN :matchIds")
    java.util.List<com.example.data_service.entity.GameScore> findAllByMatchIdIn(@Param("matchIds") java.util.List<Integer> matchIds);

    @Query(value = """
            SELECT t.name, g.map, COUNT(g.map)
            FROM esports_game_scores AS g
            INNER JOIN esports_matches AS m ON g.match_id = m.id
            INNER JOIN esports_tournaments AS t ON m.tournament_id = t.id
            WHERE t.tier = 'VCT' AND EXTRACT(YEAR FROM t.end_date) = 2026
            GROUP BY t.name, g.map
            ORDER BY t.name, COUNT(g.map) DESC
            """, nativeQuery = true)
    java.util.List<Object[]> findTournamentMapStats();
}
