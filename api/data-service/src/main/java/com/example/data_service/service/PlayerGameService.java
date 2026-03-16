package com.example.data_service.service;

import com.example.data_service.dto.PlayerGameDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.PlayerGameRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class PlayerGameService {
    private final PlayerGameRepository repo;

    public PlayerGameService(PlayerGameRepository repo) { this.repo = repo; }

    public Page<PlayerGameDto> getPlayerGames(int page, int size) { return repo.findAllAsDto(PageRequest.of(page, size)); }

    public PlayerGameDto getPlayerGame(Integer id) {
        PlayerGameDto dto = repo.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("PlayerGame not found with id: " + id);
        return dto;
    }
}
