package com.example.data_service.repository;

import com.example.data_service.dto.MapVetoDto;
import com.example.data_service.entity.MapVeto;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface MapVetoRepository extends JpaRepository<MapVeto, Integer> {
    @Query("select new com.example.data_service.dto.MapVetoDto(m.id, m.matchId, m.type, m.teamId, m.mapSelected, m.turn) from MapVeto m")
    Page<MapVetoDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.MapVetoDto(m.id, m.matchId, m.type, m.teamId, m.mapSelected, m.turn) from MapVeto m where m.id = :id")
    MapVetoDto findDtoById(@Param("id") Integer id);

    @Query("SELECT m FROM MapVeto m WHERE m.matchId IN :matchIds")
    java.util.List<com.example.data_service.entity.MapVeto> findAllByMatchIdIn(@Param("matchIds") java.util.List<Integer> matchIds);

    @Query("SELECT m FROM MapVeto m LEFT JOIN FETCH m.teamEntity LEFT JOIN FETCH m.matchEntity WHERE m.id = :id")
    java.util.Optional<com.example.data_service.entity.MapVeto> findWithRelationsById(@Param("id") Integer id);

    @Query("SELECT DISTINCT m FROM MapVeto m LEFT JOIN FETCH m.teamEntity LEFT JOIN FETCH m.matchEntity WHERE m.id IN :ids")
    java.util.List<com.example.data_service.entity.MapVeto> findAllWithRelationsByIdIn(@Param("ids") java.util.List<Integer> ids);
}
