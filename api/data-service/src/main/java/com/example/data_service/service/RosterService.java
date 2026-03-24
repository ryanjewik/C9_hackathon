package com.example.data_service.service;

import com.example.data_service.dto.RosterDto;
import com.example.data_service.dto.PlayerDto;
import com.example.data_service.dto.TeamDto;
import com.example.data_service.entity.Roster;
import com.example.data_service.entity.Player;
import com.example.data_service.entity.Team;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.RosterRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.PageImpl;
import org.springframework.stereotype.Service;

@Service
public class RosterService {
    private final RosterRepository repo;

    public RosterService(RosterRepository repo) { this.repo = repo; }

    public Page<RosterDto> getRosters(int page, int size) {
        org.springframework.data.domain.Pageable pageable = PageRequest.of(page, size);
        Page<Roster> rosterPage = repo.findAll(pageable);

        java.util.List<Integer> ids = rosterPage.getContent().stream().map(Roster::getId).collect(Collectors.toList());
        if (ids.isEmpty()) {
            return new PageImpl<>(java.util.Collections.emptyList(), pageable, rosterPage.getTotalElements());
        }

        java.util.List<Roster> full = repo.findAllWithPlayersByIdIn(ids);
        java.util.Map<Integer, Roster> map = full.stream().collect(Collectors.toMap(Roster::getId, r -> r));

        java.util.List<RosterDto> dtos = rosterPage.getContent().stream().map(r -> {
            Roster populated = map.getOrDefault(r.getId(), r);
            RosterDto rd = new RosterDto(populated.getId(), populated.getDateCreated(), populated.getMapWins(), populated.getMapLosses());
            // set per-player object fields for explicit player1..player5 keys
            Player p1 = populated.getPlayer1Entity(); if (p1 != null) rd.setPlayer1Obj(new PlayerDto(p1.getId(), p1.getNickname()));
            Player p2 = populated.getPlayer2Entity(); if (p2 != null) rd.setPlayer2Obj(new PlayerDto(p2.getId(), p2.getNickname()));
            Player p3 = populated.getPlayer3Entity(); if (p3 != null) rd.setPlayer3Obj(new PlayerDto(p3.getId(), p3.getNickname()));
            Player p4 = populated.getPlayer4Entity(); if (p4 != null) rd.setPlayer4Obj(new PlayerDto(p4.getId(), p4.getNickname()));
            Player p5 = populated.getPlayer5Entity(); if (p5 != null) rd.setPlayer5Obj(new PlayerDto(p5.getId(), p5.getNickname()));
            Team tm = populated.getTeamEntity(); if (tm != null) rd.setTeamObj(new TeamDto(tm.getId(), tm.getName(), tm.getTeamTag()));

            return rd;
        }).collect(Collectors.toList());

        return new PageImpl<>(dtos, pageable, rosterPage.getTotalElements());
    }

    public RosterDto getRoster(Integer id) {
        Optional<Roster> opt = repo.findWithPlayersById(id);
        Roster roster = opt.orElseThrow(() -> new ResourceNotFoundException("Roster not found with id: " + id));

        RosterDto rd = new RosterDto(roster.getId(), roster.getDateCreated(), roster.getMapWins(), roster.getMapLosses());
        Player p1 = roster.getPlayer1Entity(); if (p1 != null) rd.setPlayer1Obj(new PlayerDto(p1.getId(), p1.getNickname()));
        Player p2 = roster.getPlayer2Entity(); if (p2 != null) rd.setPlayer2Obj(new PlayerDto(p2.getId(), p2.getNickname()));
        Player p3 = roster.getPlayer3Entity(); if (p3 != null) rd.setPlayer3Obj(new PlayerDto(p3.getId(), p3.getNickname()));
        Player p4 = roster.getPlayer4Entity(); if (p4 != null) rd.setPlayer4Obj(new PlayerDto(p4.getId(), p4.getNickname()));
        Player p5 = roster.getPlayer5Entity(); if (p5 != null) rd.setPlayer5Obj(new PlayerDto(p5.getId(), p5.getNickname()));
        Team tm = roster.getTeamEntity(); if (tm != null) rd.setTeamObj(new TeamDto(tm.getId(), tm.getName(), tm.getTeamTag()));

        return rd;
    }
}
