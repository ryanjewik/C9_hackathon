package com.example.data_service.controller;

import com.example.data_service.dto.TournamentPlacementDto;
import com.example.data_service.service.TournamentPlacementService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/tournament-placements")
public class TournamentPlacementController {
    private static final Logger log = LoggerFactory.getLogger(TournamentPlacementController.class);
    private final TournamentPlacementService service;

    public TournamentPlacementController(TournamentPlacementService service) { this.service = service; }

    @GetMapping
    public Page<TournamentPlacementDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        log.info("GET /api/tournament-placements page={} size={}", page, size);
        return service.getPlacements(page, size);
    }

    @GetMapping("/{id}")
    public TournamentPlacementDto get(@PathVariable Integer id) {
        log.info("GET /api/tournament-placements/{}", id);
        return service.getPlacement(id);
    }
}
