package com.example.data_service.controller;

import com.example.data_service.dto.RosterDto;
import com.example.data_service.service.RosterService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/rosters")
public class RosterController {
    private final RosterService service;

    public RosterController(RosterService service) { this.service = service; }

    @GetMapping
    public Page<RosterDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        return service.getRosters(page, size);
    }

    @GetMapping("/{id}")
    public RosterDto get(@PathVariable Integer id) { return service.getRoster(id); }
}
