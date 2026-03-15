package com.example.data_service.service;

import com.example.data_service.entity.Player;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.PlayerRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;


//services layer handles business logic, calls the respository, throws exceptions, caching, preparing data
//for caching we may make another service that handles caching
@Service
public class PlayerService {
    private final PlayerRepository playerRepository;

    public PlayerService(PlayerRepository playerRepository) {
        this.playerRepository = playerRepository;
    }

    public Page<Player> getPlayers(int page, int size) {
        return playerRepository.findAll(PageRequest.of(page, size));
    }

    public Player getPlayer(Integer id) {
        return playerRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("Player not found with id: " + id));
    }
}