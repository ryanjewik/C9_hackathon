package com.example.data_service.service;

import com.example.data_service.dto.TournamentMapStatsDto;
import com.example.data_service.repository.GameScoreRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class TournamentMapStatsService {
    @Autowired
    private GameScoreRepository repository;

    public List<TournamentMapStatsDto> getTournamentMapStats() {
        return repository.findTournamentMapStats().stream()
            .map(row -> new TournamentMapStatsDto(
                (String) row[0],
                (String) row[1],
                row[2] != null ? ((Number) row[2]).longValue() : null
            ))
            .collect(Collectors.toList());
    }
}
