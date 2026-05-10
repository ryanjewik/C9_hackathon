package com.example.data_service.controller;

import com.example.data_service.dto.AgentPickRateDto;
import com.example.data_service.dto.AgentStatsDto;
import com.example.data_service.dto.PlayerStatsDto;
import com.example.data_service.service.PlayerStatsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/dashboard")
public class PlayerStatsController {
    @Autowired
    private PlayerStatsService service;

    @GetMapping("/player_stats")
    public ResponseEntity<List<PlayerStatsDto>> getTopPlayerStats(
            @RequestParam(defaultValue = "top") String sort) {
        List<PlayerStatsDto> result = "bottom".equalsIgnoreCase(sort)
            ? service.getBottomPlayerStats()
            : service.getTopPlayerStats();
        return ResponseEntity.ok(result);
    }

    @GetMapping("/agent_stats")
    public ResponseEntity<List<AgentStatsDto>> getPlayerPerAgent(
            @RequestParam(defaultValue = "top") String sort) {
        List<AgentStatsDto> result = "bottom".equalsIgnoreCase(sort)
            ? service.getBottomPlayerPerAgent()
            : service.getTopPlayerPerAgent();
        return ResponseEntity.ok(result);
    }

    @GetMapping("/agent_pickrates")
    public ResponseEntity<List<AgentPickRateDto>> getAgentPickRates() {
        return ResponseEntity.ok(service.getAgentPickRates());
    }
}
