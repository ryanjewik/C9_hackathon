package com.example.data_service.controller;

import com.example.data_service.dto.TeamDto;
import com.example.data_service.service.TeamService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/teams")
public class TeamController {
    private final TeamService service;

    public TeamController(TeamService service) { this.service = service; }

    @GetMapping
    public Page<TeamDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        return service.getTeams(page, size);
    }

    @GetMapping("/{id}")
    public TeamDto get(@PathVariable Integer id) { return service.getTeam(id); }
}
