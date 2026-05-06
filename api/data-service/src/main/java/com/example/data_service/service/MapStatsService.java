package com.example.data_service.service;

import com.example.data_service.dto.MapStatsDto;
import com.example.data_service.repository.MapVetoRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class MapStatsService {
    @Autowired
    private MapVetoRepository repository;

    public List<MapStatsDto> getMapStatsByTeamId(Integer teamId) {
        return repository.findMapStatsByTeamId(teamId).stream()
            .map(row -> new MapStatsDto(
                (String) row[0],
                ((Number) row[1]).longValue(),
                ((Number) row[2]).longValue()
            ))
            .collect(Collectors.toList());
    }
}
