package com.example.data_service.service;

import com.example.data_service.dto.TeamDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.TeamRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class TeamService {
    private final TeamRepository repo;

    public TeamService(TeamRepository repo) { this.repo = repo; }

    public Page<TeamDto> getTeams(int page, int size) { return repo.findAllAsDto(PageRequest.of(page, size)); }

    public TeamDto getTeam(Integer id) {
        TeamDto dto = repo.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("Team not found with id: " + id);
        return dto;
    }
}
