package com.example.data_service.controller;

import com.example.data_service.entity.Player;
import com.example.data_service.service.PlayerService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/players")
public class PlayerController {
    private final PlayerService playerService;

    public PlayerController(PlayerService playerService) {
        this.playerService = playerService;
    }

    @GetMapping
    public Page<Player> getPlayers(
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "20") int size
    ) {
        return playerService.getPlayers(page, size);
    }

    @GetMapping("/{id}")
    public Player getPlayer(@PathVariable Integer id) {
        return playerService.getPlayer(id);
    }
}