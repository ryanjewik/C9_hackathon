package com.example.data_service.controller;

import com.example.data_service.dto.PlayerGameDto;
import com.example.data_service.service.PlayerGameService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/player-games")
public class PlayerGameController {
    private final PlayerGameService service;

    public PlayerGameController(PlayerGameService service) { this.service = service; }

    @GetMapping
    public Page<PlayerGameDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        return service.getPlayerGames(page, size);
    }

    @GetMapping("/{id}")
    public PlayerGameDto get(@PathVariable Integer id) { return service.getPlayerGame(id); }
}
