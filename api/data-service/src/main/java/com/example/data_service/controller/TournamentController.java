package com.example.data_service.controller;

import com.example.data_service.dto.TournamentDto;
import com.example.data_service.service.TournamentService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RestController;


@RestController
@RequestMapping("/api/tournaments")
public class TournamentController {
    private static final Logger log = LoggerFactory.getLogger(TournamentController.class);
    private final TournamentService tournamentService;

    public TournamentController(TournamentService tournamentService) {
        this.tournamentService = tournamentService;
    }

    @GetMapping
    public Page<TournamentDto> getTournaments(
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "20") int size
    ) {
        log.info("GET /api/tournaments page={} size={}", page, size);
        return tournamentService.getTournaments(page, size);
    }

    @GetMapping("/{id}")
    public TournamentDto getTournament(@PathVariable Integer id) {
        log.info("GET /api/tournaments/{}", id);
        return tournamentService.getTournament(id);
    }

}