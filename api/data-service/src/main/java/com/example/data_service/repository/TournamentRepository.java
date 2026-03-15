package com.example.data_service.repository;

import com.example.data_service.entity.Tournament;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TournamentRepository extends JpaRepository<Tournament, Integer> {
}