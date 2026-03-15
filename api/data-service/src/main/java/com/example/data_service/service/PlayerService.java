package com.example.data_service.service;

import com.example.data_service.dto.PlayerDto;
import com.example.data_service.entity.Player;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.PlayerRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;


// services layer handles business logic, calls the repository, throws exceptions, caching, preparing data
// for caching we may make another service that handles caching
@Service
public class PlayerService {
    private final PlayerRepository playerRepository;

    public PlayerService(PlayerRepository playerRepository) {
        this.playerRepository = playerRepository;
    }

    public Page<PlayerDto> getPlayers(int page, int size) {
        return playerRepository.findAllAsDto(PageRequest.of(page, size));
    }

    public PlayerDto getPlayer(Integer id) {
        PlayerDto dto = playerRepository.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("Player not found with id: " + id);
        return dto;
    }
}