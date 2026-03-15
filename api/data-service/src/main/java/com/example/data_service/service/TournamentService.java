package com.example.data_service.service;

import com.example.data_service.dto.TournamentDto;
import com.example.data_service.entity.Tournament;
import com.example.data_service.exception.ResourceNotFoundException;
import com.example.data_service.repository.TournamentRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;


@Service
public class TournamentService {
    private final TournamentRepository tournamentRepository;

    public TournamentService(TournamentRepository tournamentRepository) {
        this.tournamentRepository = tournamentRepository;
    }

    public Page<TournamentDto> getTournaments(int page, int size) {
        return tournamentRepository.findAllAsDto(PageRequest.of(page, size));
    }

    public TournamentDto getTournament(Integer id) {
        TournamentDto dto = tournamentRepository.findDtoById(id);
        if (dto == null) throw new ResourceNotFoundException("Tournament not found with id: " + id);
        return dto;
    }
}