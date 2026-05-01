package com.example.data_service.service;

import com.example.data_service.dto.TournamentDto;
import com.example.data_service.entity.Tournament;
import com.example.data_service.repository.OngoingTournamentsRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.stream.Collectors;
import java.util.List;

@Service
public class OngoingTournamentsService {
    @Autowired
    private OngoingTournamentsRepository repository;

    public List<TournamentDto> getOngoingTournaments() {
        return repository.findOngoingTournaments().stream()
        .map(entity -> new TournamentDto(entity.getId(), entity.getName(), entity.getTier(), entity.getStartDate(), entity.getEndDate(), entity.getLocation(), entity.getPrizePool(), entity.getStatus()))
        .collect(Collectors.toList());
    }
}
