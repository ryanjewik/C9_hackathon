package com.example.data_service.service;

import com.example.data_service.dto.TournamentPlacementDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.TournamentPlacementRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class TournamentPlacementService {
    private final TournamentPlacementRepository repo;

    public TournamentPlacementService(TournamentPlacementRepository repo) { this.repo = repo; }

    public Page<TournamentPlacementDto> getPlacements(int page, int size) { return repo.findAllAsDto(PageRequest.of(page, size)); }

    public TournamentPlacementDto getPlacement(Integer id) {
        TournamentPlacementDto dto = repo.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("TournamentPlacement not found with id: " + id);
        return dto;
    }
}
