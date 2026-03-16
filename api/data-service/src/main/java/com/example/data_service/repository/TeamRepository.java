package com.example.data_service.repository;

import com.example.data_service.dto.TeamDto;
import com.example.data_service.entity.Team;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface TeamRepository extends JpaRepository<Team, Integer> {
    @Query("select new com.example.data_service.dto.TeamDto(t.id, t.name, t.teamTag, t.location, t.titles, t.matchWins, t.matchLosses, t.currentRosterId) from Team t")
    Page<TeamDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.TeamDto(t.id, t.name, t.teamTag, t.location, t.titles, t.matchWins, t.matchLosses, t.currentRosterId) from Team t where t.id = :id")
    TeamDto findDtoById(@Param("id") Integer id);
}
