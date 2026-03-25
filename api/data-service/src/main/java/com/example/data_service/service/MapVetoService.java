package com.example.data_service.service;

import com.example.data_service.dto.MapVetoDto;
import com.example.data_service.dto.TeamDto;
import com.example.data_service.dto.MatchSummaryDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.MapVetoRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class MapVetoService {
    private final MapVetoRepository repo;

    public MapVetoService(MapVetoRepository repo) { this.repo = repo; }

    public Page<MapVetoDto> getMapVetos(int page, int size) {
        org.springframework.data.domain.Pageable pageable = PageRequest.of(page, size);
        Page<com.example.data_service.entity.MapVeto> pageEnt = repo.findAll(pageable);

        java.util.List<Integer> ids = pageEnt.getContent().stream().map(com.example.data_service.entity.MapVeto::getId).collect(java.util.stream.Collectors.toList());
        if (ids.isEmpty()) return Page.empty(pageable);

        java.util.List<com.example.data_service.entity.MapVeto> full = repo.findAllWithRelationsByIdIn(ids);
        java.util.Map<Integer, com.example.data_service.entity.MapVeto> map = full.stream().collect(java.util.stream.Collectors.toMap(com.example.data_service.entity.MapVeto::getId, m -> m));

        java.util.List<MapVetoDto> dtos = pageEnt.getContent().stream().map(m -> {
            com.example.data_service.entity.MapVeto populated = map.getOrDefault(m.getId(), m);
            MapVetoDto dto = new MapVetoDto(populated.getId(), populated.getMatchId(), populated.getType(), populated.getTeamId(), populated.getMapSelected(), populated.getTurn());
            com.example.data_service.entity.Team team = populated.getTeamEntity(); if (team != null) dto.setTeamObj(new TeamDto(team.getId(), team.getName(), team.getTeamTag()));
            com.example.data_service.entity.Match match = populated.getMatchEntity(); if (match != null) dto.setMatchObj(new MatchSummaryDto(match.getTeam1Id(), match.getTeam1Name(), match.getTeam2Id(), match.getTeam2Name()));
            return dto;
        }).collect(java.util.stream.Collectors.toList());

        return new org.springframework.data.domain.PageImpl<>(dtos, pageable, pageEnt.getTotalElements());
    }

    public MapVetoDto getMapVeto(Integer id) {
        java.util.Optional<com.example.data_service.entity.MapVeto> opt = repo.findWithRelationsById(id);
        com.example.data_service.entity.MapVeto mv = opt.orElseThrow(() -> new ResourceNotFoundException("MapVeto not found with id: " + id));

        MapVetoDto dto = new MapVetoDto(mv.getId(), mv.getMatchId(), mv.getType(), mv.getTeamId(), mv.getMapSelected(), mv.getTurn());
        com.example.data_service.entity.Team team = mv.getTeamEntity(); if (team != null) dto.setTeamObj(new TeamDto(team.getId(), team.getName(), team.getTeamTag()));
        com.example.data_service.entity.Match match = mv.getMatchEntity(); if (match != null) dto.setMatchObj(new MatchSummaryDto(match.getTeam1Id(), match.getTeam1Name(), match.getTeam2Id(), match.getTeam2Name()));

        return dto;
    }
}
