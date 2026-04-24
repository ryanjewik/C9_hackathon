package com.example.data_service.controller;

import com.example.data_service.dto.PlayerDto;
import com.example.data_service.service.PlayerService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RestController;


// controllers handle auth sometimes (will have gateway do that?), call the service, call the dto, and return response codes
@RestController
@RequestMapping("/api/players")
public class PlayerController {
    private static final Logger log = LoggerFactory.getLogger(PlayerController.class);
    private final PlayerService playerService;

    public PlayerController(PlayerService playerService) {
        this.playerService = playerService;
    }

    @GetMapping
    public Page<PlayerDto> getPlayers(
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "20") int size
    ) {
        log.info("GET /api/players page={} size={}", page, size);
        return playerService.getPlayers(page, size);
    }

    @GetMapping("/{id}")
    public PlayerDto getPlayer(@PathVariable Integer id) {
        log.info("GET /api/players/{}", id);
        return playerService.getPlayer(id);
    }
}