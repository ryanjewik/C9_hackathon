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
}
