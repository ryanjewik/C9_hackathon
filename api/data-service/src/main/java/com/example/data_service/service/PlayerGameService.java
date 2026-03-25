package com.example.data_service.service;

import com.example.data_service.dto.PlayerGameDto;
import com.example.data_service.dto.PlayerDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.PlayerGameRepository;
import com.example.data_service.repository.PlayerRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class PlayerGameService {
    private final PlayerGameRepository repo;
    private final PlayerRepository playerRepo;

    public PlayerGameService(PlayerGameRepository repo, PlayerRepository playerRepo) { this.repo = repo; this.playerRepo = playerRepo; }

    public Page<PlayerGameDto> getPlayerGames(int page, int size) {
        org.springframework.data.domain.Pageable pageable = PageRequest.of(page, size);
        Page<com.example.data_service.entity.PlayerGame> pageEnt = repo.findAll(pageable);

        java.util.List<Integer> pids = pageEnt.getContent().stream().map(com.example.data_service.entity.PlayerGame::getPlayerId).collect(java.util.stream.Collectors.toList());
            final java.util.Map<Integer, PlayerDto> pmap;
            if (pids.isEmpty()) {
                pmap = java.util.Collections.emptyMap();
            } else {
                java.util.List<PlayerDto> players = playerRepo.findAllDtoByIdIn(pids);
                pmap = players.stream().collect(java.util.stream.Collectors.toMap(PlayerDto::getId, p -> p));
        }

        java.util.List<PlayerGameDto> dtos = pageEnt.getContent().stream().map(pg -> {
            PlayerGameDto dto = new PlayerGameDto(pg.getId(), pg.getMatchId(), pg.getGameId(), pg.getPlayerId(), pg.getTeamId(), pg.getRosterId(), pg.getTournamentId(), pg.getMap(), pg.getAgent(), pg.getRating(), pg.getAcs(), pg.getKills(), pg.getDeaths(), pg.getAssists(), pg.getKast(), pg.getAdr(), pg.getHsPercent(), pg.getFk(), pg.getFd());
            PlayerDto pd = pmap.get(pg.getPlayerId()); if (pd != null) dto.setPlayerObj(pd);
            return dto;
        }).collect(java.util.stream.Collectors.toList());

        return new org.springframework.data.domain.PageImpl<>(dtos, pageable, pageEnt.getTotalElements());
    }

    public PlayerGameDto getPlayerGame(Integer id) {
        java.util.Optional<com.example.data_service.entity.PlayerGame> opt = repo.findById(id);
        com.example.data_service.entity.PlayerGame pg = opt.orElseThrow(() -> new ResourceNotFoundException("PlayerGame not found with id: " + id));

        PlayerGameDto dto = new PlayerGameDto(pg.getId(), pg.getMatchId(), pg.getGameId(), pg.getPlayerId(), pg.getTeamId(), pg.getRosterId(), pg.getTournamentId(), pg.getMap(), pg.getAgent(), pg.getRating(), pg.getAcs(), pg.getKills(), pg.getDeaths(), pg.getAssists(), pg.getKast(), pg.getAdr(), pg.getHsPercent(), pg.getFk(), pg.getFd());
        java.util.List<PlayerDto> pdList = playerRepo.findAllDtoByIdIn(java.util.List.of(pg.getPlayerId()));
        if (pdList != null && !pdList.isEmpty()) dto.setPlayerObj(pdList.get(0));
        return dto;
    }
}
