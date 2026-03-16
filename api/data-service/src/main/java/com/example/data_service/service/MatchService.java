package com.example.data_service.service;

import com.example.data_service.dto.MatchDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.MatchRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class MatchService {
    private final MatchRepository repo;

    public MatchService(MatchRepository repo) { this.repo = repo; }

    public Page<MatchDto> getMatches(int page, int size) { return repo.findAllAsDto(PageRequest.of(page, size)); }

    public MatchDto getMatch(Integer id) {
        MatchDto dto = repo.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("Match not found with id: " + id);
        return dto;
    }
}
