package com.example.data_service.service;

import com.example.data_service.dto.GameScoreDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.GameScoreRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class GameScoreService {
    private final GameScoreRepository repo;

    public GameScoreService(GameScoreRepository repo) { this.repo = repo; }

    public Page<GameScoreDto> getGameScores(int page, int size) { return repo.findAllAsDto(PageRequest.of(page, size)); }

    public GameScoreDto getGameScore(Integer id) {
        GameScoreDto dto = repo.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("GameScore not found with id: " + id);
        return dto;
    }
}
