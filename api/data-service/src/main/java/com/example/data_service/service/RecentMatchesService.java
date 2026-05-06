package com.example.data_service.service;

import com.example.data_service.dto.MatchDto;
import com.example.data_service.repository.RecentMatchesRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class RecentMatchesService {
    @Autowired
    private RecentMatchesRepository repository;

    public List<MatchDto> getRecentMatches() {
        return repository.findRecentMatches().stream()
            .map(entity -> new MatchDto(
                entity.getId(),
                entity.getPhase(),
                entity.getDate(),
                entity.getPatch(),
                entity.getTournamentId(),
                entity.getTournamentName(),
                entity.getTeam1Name(),
                entity.getTeam1Id(),
                entity.getTeam1Score(),
                entity.getTeam2Name(),
                entity.getTeam2Id(),
                entity.getTeam2Score(),
                entity.getWinner(),
                entity.getFormat(),
                entity.getMap1(),
                entity.getMap2(),
                entity.getMap3(),
                entity.getMap4(),
                entity.getMap5()
            ))
            .collect(Collectors.toList());
    }
}
