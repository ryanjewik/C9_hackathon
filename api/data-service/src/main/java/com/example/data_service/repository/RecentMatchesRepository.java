package com.example.data_service.repository;

import com.example.data_service.entity.Match;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface RecentMatchesRepository extends JpaRepository<Match, Integer> {
    @Query(value = "SELECT m.* FROM esports_matches AS m INNER JOIN esports_tournaments AS t ON t.id = m.tournament_id WHERE t.tier = 'VCT' ORDER BY m.date DESC LIMIT 3",
    nativeQuery = true)
    List<Match> findRecentMatches();
}
