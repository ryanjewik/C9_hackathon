package com.example.data_service.service;

import com.example.data_service.dto.MatchDto;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.MatchRepository;
import com.example.data_service.repository.MapVetoRepository;
import com.example.data_service.repository.GameScoreRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

@Service
public class MatchService {
    private final MatchRepository repo;
    private final MapVetoRepository mapVetoRepo;
    private final GameScoreRepository gameScoreRepo;

    public MatchService(MatchRepository repo, MapVetoRepository mapVetoRepo, GameScoreRepository gameScoreRepo) { this.repo = repo; this.mapVetoRepo = mapVetoRepo; this.gameScoreRepo = gameScoreRepo; }

    public Page<MatchDto> getMatches(int page, int size) {
        org.springframework.data.domain.Pageable pageable = PageRequest.of(page, size);
        Page<com.example.data_service.entity.Match> pageEnt = repo.findAll(pageable);

        java.util.List<Integer> ids = pageEnt.getContent().stream().map(com.example.data_service.entity.Match::getId).collect(java.util.stream.Collectors.toList());
        if (ids.isEmpty()) return Page.empty(pageable);

        java.util.List<com.example.data_service.entity.MapVeto> vetos = mapVetoRepo.findAllByMatchIdIn(ids);
        java.util.Map<Integer, java.util.List<Integer>> vetoMap = vetos.stream().collect(java.util.stream.Collectors.groupingBy(com.example.data_service.entity.MapVeto::getMatchId, java.util.stream.Collectors.mapping(com.example.data_service.entity.MapVeto::getId, java.util.stream.Collectors.toList())));

        java.util.List<com.example.data_service.entity.GameScore> scores = gameScoreRepo.findAllByMatchIdIn(ids);
        java.util.Map<Integer, java.util.List<Integer>> scoreMap = scores.stream().collect(java.util.stream.Collectors.groupingBy(com.example.data_service.entity.GameScore::getMatchId, java.util.stream.Collectors.mapping(com.example.data_service.entity.GameScore::getId, java.util.stream.Collectors.toList())));

        java.util.List<MatchDto> dtos = pageEnt.getContent().stream().map(m -> {
            MatchDto dto = new MatchDto(m.getId(), m.getPhase(), m.getDate(), m.getPatch(), m.getTournamentId(), m.getTournamentName(), m.getTeam1Name(), m.getTeam1Id(), m.getTeam1Score(), m.getTeam2Name(), m.getTeam2Id(), m.getTeam2Score(), m.getWinner(), m.getFormat(), m.getMap1(), m.getMap2(), m.getMap3(), m.getMap4(), m.getMap5());
            dto.setMapVetoIds(vetoMap.getOrDefault(m.getId(), java.util.Collections.emptyList()));
            dto.setGameScoreIds(scoreMap.getOrDefault(m.getId(), java.util.Collections.emptyList()));
            return dto;
        }).collect(java.util.stream.Collectors.toList());

        return new org.springframework.data.domain.PageImpl<>(dtos, pageable, pageEnt.getTotalElements());
    }

    public MatchDto getMatch(Integer id) {
        java.util.Optional<com.example.data_service.entity.Match> opt = repo.findById(id);
        com.example.data_service.entity.Match m = opt.orElseThrow(() -> new ResourceNotFoundException("Match not found with id: " + id));

        MatchDto dto = new MatchDto(m.getId(), m.getPhase(), m.getDate(), m.getPatch(), m.getTournamentId(), m.getTournamentName(), m.getTeam1Name(), m.getTeam1Id(), m.getTeam1Score(), m.getTeam2Name(), m.getTeam2Id(), m.getTeam2Score(), m.getWinner(), m.getFormat(), m.getMap1(), m.getMap2(), m.getMap3(), m.getMap4(), m.getMap5());

        java.util.List<com.example.data_service.entity.MapVeto> vetos = mapVetoRepo.findAllByMatchIdIn(java.util.List.of(m.getId()));
        dto.setMapVetoIds(vetos.stream().map(com.example.data_service.entity.MapVeto::getId).collect(java.util.stream.Collectors.toList()));

        java.util.List<com.example.data_service.entity.GameScore> scores = gameScoreRepo.findAllByMatchIdIn(java.util.List.of(m.getId()));
        dto.setGameScoreIds(scores.stream().map(com.example.data_service.entity.GameScore::getId).collect(java.util.stream.Collectors.toList()));

        return dto;
    }
}
