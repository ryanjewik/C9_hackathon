package com.example.data_service.controller;

import com.example.data_service.entity.Tournament;
import com.example.data_service.service.TournamentService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;


@RestController

@RequestMapping("/api/tournaments")
public class TournamentController {
    private final TournamentService tournamentService;

    public TournamentController(TournamentService tournamentService) {
        this.tournamentService = tournamentService;
    }

    @GetMapping
    public Page<Tournament> getTournaments(
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "20") int size
    ) {
        return tournamentService.getTournaments(page, size);
    }

    @GetMapping("/{id}")
    public Tournament getTournament(@PathVariable Integer id) {
        return playerService.getPlayer(id);
    }

}