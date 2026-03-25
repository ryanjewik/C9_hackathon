package com.example.data_service.service;

import com.example.data_service.dto.TournamentPlacementDto;
import com.example.data_service.dto.PlayerDto;
import com.example.data_service.dto.TeamDto;
import com.example.data_service.dto.TournamentDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.TournamentPlacementRepository;
import com.example.data_service.repository.PlayerRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class TournamentPlacementService {
    private final TournamentPlacementRepository repo;
    private final PlayerRepository playerRepo;

    public TournamentPlacementService(TournamentPlacementRepository repo, PlayerRepository playerRepo) { this.repo = repo; this.playerRepo = playerRepo; }

    public Page<TournamentPlacementDto> getPlacements(int page, int size) {
        org.springframework.data.domain.Pageable pageable = PageRequest.of(page, size);
        Page<com.example.data_service.entity.TournamentPlacement> pageEnt = repo.findAll(pageable);

        java.util.List<Integer> ids = pageEnt.getContent().stream().map(com.example.data_service.entity.TournamentPlacement::getId).collect(java.util.stream.Collectors.toList());
        if (ids.isEmpty()) return Page.empty(pageable);

        java.util.List<com.example.data_service.entity.TournamentPlacement> full = repo.findAllWithRelationsByIdIn(ids);
        java.util.Map<Integer, com.example.data_service.entity.TournamentPlacement> map = full.stream().collect(java.util.stream.Collectors.toMap(com.example.data_service.entity.TournamentPlacement::getId, p -> p));

        java.util.List<TournamentPlacementDto> dtos = pageEnt.getContent().stream().map(tp -> {
            com.example.data_service.entity.TournamentPlacement populated = map.getOrDefault(tp.getId(), tp);
            TournamentPlacementDto dto = new TournamentPlacementDto(populated.getId(), populated.getPlacement(), populated.getPrizeMoney(), populated.getStage());

            // team
            com.example.data_service.entity.Team team = populated.getTeamEntity();
            if (team != null) dto.setTeamObj(new TeamDto(team.getId(), team.getName(), team.getTeamTag()));

            // tournament
            com.example.data_service.entity.Tournament tour = populated.getTournamentEntity();
            if (tour != null) dto.setTournamentObj(new TournamentDto(tour.getId(), tour.getName()));

            // players: fetch lightweight PlayerDto and preserve ordering
            java.util.List<Integer> pids = populated.getPlayers();
            if (pids != null && !pids.isEmpty()) {
                java.util.List<PlayerDto> players = playerRepo.findAllDtoByIdIn(pids);
                java.util.Map<Integer, PlayerDto> pmap = players.stream().collect(java.util.stream.Collectors.toMap(PlayerDto::getId, p -> p));
                java.util.List<PlayerDto> ordered = new java.util.ArrayList<>();
                for (Integer id : pids) {
                    PlayerDto pd = pmap.get(id);
                    if (pd != null) ordered.add(pd);
                }
                dto.setPlayersObj(ordered);
            }

            return dto;
        }).collect(java.util.stream.Collectors.toList());

        return new org.springframework.data.domain.PageImpl<>(dtos, pageable, pageEnt.getTotalElements());
    }

    public TournamentPlacementDto getPlacement(Integer id) {
        java.util.Optional<com.example.data_service.entity.TournamentPlacement> opt = repo.findWithRelationsById(id);
        com.example.data_service.entity.TournamentPlacement tp = opt.orElseThrow(() -> new ResourceNotFoundException("TournamentPlacement not found with id: " + id));

        TournamentPlacementDto dto = new TournamentPlacementDto(tp.getId(), tp.getPlacement(), tp.getPrizeMoney(), tp.getStage());
        com.example.data_service.entity.Team team = tp.getTeamEntity(); if (team != null) dto.setTeamObj(new TeamDto(team.getId(), team.getName(), team.getTeamTag()));
        com.example.data_service.entity.Tournament tour = tp.getTournamentEntity(); if (tour != null) dto.setTournamentObj(new TournamentDto(tour.getId(), tour.getName()));

        java.util.List<Integer> pids = tp.getPlayers();
        if (pids != null && !pids.isEmpty()) {
            java.util.List<PlayerDto> players = playerRepo.findAllDtoByIdIn(pids);
            java.util.Map<Integer, PlayerDto> pmap = players.stream().collect(java.util.stream.Collectors.toMap(PlayerDto::getId, p -> p));
            java.util.List<PlayerDto> ordered = new java.util.ArrayList<>();
            for (Integer pid : pids) { PlayerDto pd = pmap.get(pid); if (pd != null) ordered.add(pd); }
            dto.setPlayersObj(ordered);
        }

        return dto;
    }
}
