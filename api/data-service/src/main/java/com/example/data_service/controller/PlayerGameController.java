package com.example.data_service.controller;

import com.example.data_service.dto.PlayerGameDto;
import com.example.data_service.service.PlayerGameService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/player-games")
public class PlayerGameController {
    private static final Logger log = LoggerFactory.getLogger(PlayerGameController.class);
    private final PlayerGameService service;

    public PlayerGameController(PlayerGameService service) { this.service = service; }

    @GetMapping
    public Page<PlayerGameDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        log.info("GET /api/player-games page={} size={}", page, size);
        return service.getPlayerGames(page, size);
    }

    @GetMapping("/{id}")
    public PlayerGameDto get(@PathVariable Integer id) {
        log.info("GET /api/player-games/{}", id);
        return service.getPlayerGame(id);
    }
}
