package com.example.data_service.repository;

import com.example.data_service.dto.RosterDto;
import com.example.data_service.entity.Roster;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface RosterRepository extends JpaRepository<Roster, Integer> {
    @Query("select new com.example.data_service.dto.RosterDto(r.id, r.teamId, r.player1, r.player2, r.player3, r.player4, r.player5, r.dateCreated, r.mapWins, r.mapLosses) from Roster r")
    Page<RosterDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.RosterDto(r.id, r.teamId, r.player1, r.player2, r.player3, r.player4, r.player5, r.dateCreated, r.mapWins, r.mapLosses) from Roster r where r.id = :id")
    RosterDto findDtoById(@Param("id") Integer id);
}
