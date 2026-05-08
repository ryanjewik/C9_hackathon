package com.example.data_service.service;

import com.example.data_service.dto.AgentPickRateDto;
import com.example.data_service.dto.AgentStatsDto;
import com.example.data_service.dto.PlayerStatsDto;
import com.example.data_service.repository.PlayerGameRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class PlayerStatsService {
    @Autowired
    private PlayerGameRepository repository;

    public List<PlayerStatsDto> getTopPlayerStats() {
        return repository.findTopPlayerStats().stream()
            .map(row -> {
                List<String> agents;
                Object agentsObj = row[1];
                // STRING_AGG returns a plain comma-separated String
                if (agentsObj instanceof String str && !str.isBlank()) {
                    agents = Arrays.asList(str.split(","));
                } else {
                    agents = List.of();
                }
                return new PlayerStatsDto(
                    (String) row[0],
                    agents,
                    row[2] != null ? ((Number) row[2]).doubleValue() : null,
                    row[3] != null ? ((Number) row[3]).longValue() : null,
                    row[4] != null ? ((Number) row[4]).longValue() : null,
                    row[5] != null ? ((Number) row[5]).longValue() : null,
                    row[6] != null ? ((Number) row[6]).longValue() : null,
                    row[7] != null ? ((Number) row[7]).longValue() : null
                );
            })
            .collect(Collectors.toList());
    }

    public List<AgentPickRateDto> getAgentPickRates() {
        return repository.findAgentPickRates().stream()
            .map(row -> new AgentPickRateDto(
                (String) row[0],
                (String) row[1],
                row[2] != null ? ((Number) row[2]).longValue() : null,
                row[3] != null ? ((Number) row[3]).longValue() : null,
                row[4] != null ? ((Number) row[4]).doubleValue() : null
            ))
            .collect(Collectors.toList());
    }

    public List<AgentStatsDto> getTopPlayerPerAgent() {
        return repository.findTopPlayerPerAgent().stream()
            .map(row -> new AgentStatsDto(
                (String) row[0],
                (String) row[1],
                row[2] != null ? ((Number) row[2]).doubleValue() : null,
                row[3] != null ? ((Number) row[3]).longValue() : null,
                row[4] != null ? ((Number) row[4]).longValue() : null,
                row[5] != null ? ((Number) row[5]).longValue() : null,
                row[6] != null ? ((Number) row[6]).longValue() : null,
                row[7] != null ? ((Number) row[7]).longValue() : null
            ))
            .collect(Collectors.toList());
    }
}
