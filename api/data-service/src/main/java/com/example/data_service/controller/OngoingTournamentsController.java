package com.example.data_service.controller;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import com.example.data_service.dto.TournamentDto;
import com.example.data_service.service.OngoingTournamentsService;
import java.util.List;

@RestController
@RequestMapping("/dashboard")
public class OngoingTournamentsController {
    @Autowired
    private OngoingTournamentsService service;

    @GetMapping("/ongoing_tournaments")
    public ResponseEntity<List<TournamentDto>> getOngoingTournaments() {
        return ResponseEntity.ok(service.getOngoingTournaments());
    }
}
