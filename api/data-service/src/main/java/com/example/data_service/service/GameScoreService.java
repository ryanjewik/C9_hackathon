package com.example.data_service.service;

import com.example.data_service.dto.GameScoreDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.GameScoreRepository;
import com.example.data_service.repository.PlayerGameRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class GameScoreService {
    private final GameScoreRepository repo;
    private final PlayerGameRepository playerGameRepo;

    public GameScoreService(GameScoreRepository repo, PlayerGameRepository playerGameRepo) { this.repo = repo; this.playerGameRepo = playerGameRepo; }

    public Page<GameScoreDto> getGameScores(int page, int size) {
        org.springframework.data.domain.Pageable pageable = PageRequest.of(page, size);
        Page<com.example.data_service.entity.GameScore> pageEnt = repo.findAll(pageable);

        java.util.List<Integer> ids = pageEnt.getContent().stream().map(com.example.data_service.entity.GameScore::getId).collect(java.util.stream.Collectors.toList());
        if (ids.isEmpty()) return Page.empty(pageable);

        java.util.List<com.example.data_service.entity.PlayerGame> pgs = playerGameRepo.findAllByGameIdIn(ids);
        java.util.Map<Integer, java.util.List<Integer>> pgMap = pgs.stream().collect(java.util.stream.Collectors.groupingBy(com.example.data_service.entity.PlayerGame::getGameId, java.util.stream.Collectors.mapping(com.example.data_service.entity.PlayerGame::getId, java.util.stream.Collectors.toList())));

        java.util.List<GameScoreDto> dtos = pageEnt.getContent().stream().map(g -> {
            GameScoreDto dto = new GameScoreDto(g.getId(), g.getMatchId(), g.getTeam1Score(), g.getTeam2Score(), g.getTeam1Id(), g.getTeam2Id(), g.getTeam1Name(), g.getTeam2Name(), g.getMap(), g.getWinner());
            dto.setPlayerGameIds(pgMap.getOrDefault(g.getId(), java.util.Collections.emptyList()));
            return dto;
        }).collect(java.util.stream.Collectors.toList());

        return new org.springframework.data.domain.PageImpl<>(dtos, pageable, pageEnt.getTotalElements());
    }

    public GameScoreDto getGameScore(Integer id) {
        java.util.Optional<com.example.data_service.entity.GameScore> opt = repo.findById(id);
        com.example.data_service.entity.GameScore g = opt.orElseThrow(() -> new ResourceNotFoundException("GameScore not found with id: " + id));

        GameScoreDto dto = new GameScoreDto(g.getId(), g.getMatchId(), g.getTeam1Score(), g.getTeam2Score(), g.getTeam1Id(), g.getTeam2Id(), g.getTeam1Name(), g.getTeam2Name(), g.getMap(), g.getWinner());

        java.util.List<com.example.data_service.entity.PlayerGame> pgs = playerGameRepo.findAllByGameIdIn(java.util.List.of(g.getId()));
        dto.setPlayerGameIds(pgs.stream().map(com.example.data_service.entity.PlayerGame::getId).collect(java.util.stream.Collectors.toList()));

        return dto;
    }
}
