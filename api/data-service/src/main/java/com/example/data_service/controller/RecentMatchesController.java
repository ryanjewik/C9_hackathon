package com.example.data_service.controller;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import com.example.data_service.dto.MatchDto;
import com.example.data_service.dto.MapStatsDto;
import com.example.data_service.dto.TeamMatchHistoryDto;
import com.example.data_service.service.RecentMatchesService;
import com.example.data_service.service.MapStatsService;
import java.util.List;

@RestController
@RequestMapping("/dashboard")
public class RecentMatchesController {
    @Autowired
    private RecentMatchesService service;

    @Autowired
    private MapStatsService mapStatsService;

    @GetMapping("/recent_matches")
    public ResponseEntity<List<MatchDto>> getRecentMatches() {
        return ResponseEntity.ok(service.getRecentMatches());
    }

    @GetMapping("/map_stats")
    public ResponseEntity<List<MapStatsDto>> getMapStats(@RequestParam Integer teamId) {
        return ResponseEntity.ok(mapStatsService.getMapStatsByTeamId(teamId));
    }

    @GetMapping("/team_match_history")
    public ResponseEntity<List<TeamMatchHistoryDto>> getTeamMatchHistory(@RequestParam Integer teamId) {
        return ResponseEntity.ok(service.getTeamMatchHistory(teamId));
    }
}
