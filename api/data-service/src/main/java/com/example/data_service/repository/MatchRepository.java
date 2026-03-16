package com.example.data_service.repository;

import com.example.data_service.dto.MatchDto;
import com.example.data_service.entity.Match;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface MatchRepository extends JpaRepository<Match, Integer> {
    @Query("select new com.example.data_service.dto.MatchDto(m.id, m.phase, m.date, m.patch, m.tournamentId, m.tournamentName, m.team1Name, m.team1Id, m.team1Score, m.team2Name, m.team2Id, m.team2Score, m.winner, m.format, m.map1, m.map2, m.map3, m.map4, m.map5) from Match m")
    Page<MatchDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.MatchDto(m.id, m.phase, m.date, m.patch, m.tournamentId, m.tournamentName, m.team1Name, m.team1Id, m.team1Score, m.team2Name, m.team2Id, m.team2Score, m.winner, m.format, m.map1, m.map2, m.map3, m.map4, m.map5) from Match m where m.id = :id")
    MatchDto findDtoById(@Param("id") Integer id);
}
