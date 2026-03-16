package com.example.data_service.controller;

import com.example.data_service.dto.MapVetoDto;
import com.example.data_service.service.MapVetoService;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/map-veto")
public class MapVetoController {
    private final MapVetoService service;

    public MapVetoController(MapVetoService service) { this.service = service; }

    @GetMapping
    public Page<MapVetoDto> list(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        return service.getMapVetos(page, size);
    }

    @GetMapping("/{id}")
    public MapVetoDto get(@PathVariable Integer id) { return service.getMapVeto(id); }
}
