package com.example.data_service.controller;

import com.example.data_service.dto.MatchDto;
import com.example.data_service.service.MatchService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/matches")
public class MatchController {
    private static final Logger log = LoggerFactory.getLogger(MatchController.class);
    private final MatchService service;

    public MatchController(MatchService service) { this.service = service; }

    @GetMapping
    public Page<MatchDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        log.info("GET /api/matches page={} size={}", page, size);
        return service.getMatches(page, size);
    }

    @GetMapping("/{id}")
    public MatchDto get(@PathVariable Integer id) {
        log.info("GET /api/matches/{}", id);
        return service.getMatch(id);
    }
}
