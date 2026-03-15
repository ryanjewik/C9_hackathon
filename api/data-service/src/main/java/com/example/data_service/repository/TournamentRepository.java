package com.example.data_service.repository;

import com.example.data_service.dto.TournamentDto;
import com.example.data_service.entity.Tournament;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface TournamentRepository extends JpaRepository<Tournament, Integer> {
	@Query("select new com.example.data_service.dto.TournamentDto(t.id, t.name, t.tier, t.startDate, t.endDate, t.location, t.prizePool, t.status) from Tournament t")
	Page<TournamentDto> findAllAsDto(Pageable pageable);

	@Query("select new com.example.data_service.dto.TournamentDto(t.id, t.name, t.tier, t.startDate, t.endDate, t.location, t.prizePool, t.status) from Tournament t where t.id = :id")
	TournamentDto findDtoById(@Param("id") Integer id);
}