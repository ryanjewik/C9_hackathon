package com.example.data_service.service;

import com.example.data_service.dto.MapVetoDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.MapVetoRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class MapVetoService {
    private final MapVetoRepository repo;

    public MapVetoService(MapVetoRepository repo) { this.repo = repo; }

    public Page<MapVetoDto> getMapVetos(int page, int size) { return repo.findAllAsDto(PageRequest.of(page, size)); }

    public MapVetoDto getMapVeto(Integer id) {
        MapVetoDto dto = repo.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("MapVeto not found with id: " + id);
        return dto;
    }
}
