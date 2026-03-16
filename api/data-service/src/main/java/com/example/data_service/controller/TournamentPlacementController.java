package com.example.data_service.controller;

import com.example.data_service.dto.TournamentPlacementDto;
import com.example.data_service.service.TournamentPlacementService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/tournament-placements")
public class TournamentPlacementController {
    private final TournamentPlacementService service;

    public TournamentPlacementController(TournamentPlacementService service) { this.service = service; }

    @GetMapping
    public Page<TournamentPlacementDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        return service.getPlacements(page, size);
    }

    @GetMapping("/{id}")
    public TournamentPlacementDto get(@PathVariable Integer id) { return service.getPlacement(id); }
}
