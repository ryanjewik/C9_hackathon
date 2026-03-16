package com.example.data_service.service;

import com.example.data_service.dto.RosterDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.RosterRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class RosterService {
    private final RosterRepository repo;

    public RosterService(RosterRepository repo) { this.repo = repo; }

    public Page<RosterDto> getRosters(int page, int size) { return repo.findAllAsDto(PageRequest.of(page, size)); }

    public RosterDto getRoster(Integer id) {
        RosterDto dto = repo.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("Roster not found with id: " + id);
        return dto;
    }
}
