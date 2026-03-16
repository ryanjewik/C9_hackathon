package com.example.data_service.repository;

import com.example.data_service.dto.TournamentPlacementDto;
import com.example.data_service.entity.TournamentPlacement;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface TournamentPlacementRepository extends JpaRepository<TournamentPlacement, Integer> {
    @Query("select new com.example.data_service.dto.TournamentPlacementDto(t.id, t.tournamentId, t.placement, t.esportsTeamId, t.prizeMoney, t.stage, t.players) from TournamentPlacement t")
    Page<TournamentPlacementDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.TournamentPlacementDto(t.id, t.tournamentId, t.placement, t.esportsTeamId, t.prizeMoney, t.stage, t.players) from TournamentPlacement t where t.id = :id")
    TournamentPlacementDto findDtoById(@Param("id") Integer id);
}
