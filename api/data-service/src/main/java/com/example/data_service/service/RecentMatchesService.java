package com.example.data_service.service;

import com.example.data_service.dto.MatchDto;
import com.example.data_service.dto.TeamMatchHistoryDto;
import com.example.data_service.repository.RecentMatchesRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
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

    public List<TeamMatchHistoryDto> getTeamMatchHistory(Integer teamId) {
        return repository.findTeamMatchHistory(teamId).stream()
            .map(row -> {
                LocalDateTime date = null;
                if (row[0] instanceof LocalDateTime ldt) {
                    date = ldt;
                } else if (row[0] instanceof java.sql.Timestamp ts) {
                    date = ts.toLocalDateTime();
                }
                return new TeamMatchHistoryDto(
                    date,
                    (Boolean) row[1],
                    (String) row[2],
                    row[3] != null ? ((Number) row[3]).intValue() : null,
                    row[4] != null ? ((Number) row[4]).intValue() : null,
                    (String) row[5]
                );
            })
            .collect(Collectors.toList());
    }
}
