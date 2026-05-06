package com.example.data_service.controller;

import com.example.data_service.dto.TournamentMapStatsDto;
import com.example.data_service.service.TournamentMapStatsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/dashboard")
public class TournamentMapStatsController {
    @Autowired
    private TournamentMapStatsService service;

    @GetMapping("/tournament_map_stats")
    public ResponseEntity<List<TournamentMapStatsDto>> getTournamentMapStats() {
        return ResponseEntity.ok(service.getTournamentMapStats());
    }
}
