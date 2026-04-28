package com.example.data_service.controller;

import com.example.data_service.dto.GameScoreDto;
import com.example.data_service.service.GameScoreService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/game-scores")
public class GameScoreController {
    private static final Logger log = LoggerFactory.getLogger(GameScoreController.class);
    private final GameScoreService service;

    public GameScoreController(GameScoreService service) { this.service = service; }

    @GetMapping
    public Page<GameScoreDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        log.info("GET /api/game-scores page={} size={}", page, size);
        return service.getGameScores(page, size);
    }

    @GetMapping("/{id}")
    public GameScoreDto get(@PathVariable Integer id) {
        log.info("GET /api/game-scores/{}", id);
        return service.getGameScore(id);
    }
}
