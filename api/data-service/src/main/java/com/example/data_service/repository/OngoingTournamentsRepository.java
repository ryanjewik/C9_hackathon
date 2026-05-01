package com.example.data_service.repository;


import com.example.data_service.entity.Tournament;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;


import java.util.List;

public interface OngoingTournamentsRepository extends JpaRepository <Tournament, Integer> {
    @Query(value = "SELECT * FROM esports_tournaments WHERE status = 'ongoing' AND tier = 'VCT'", 
    nativeQuery = true)
    List<Tournament> findOngoingTournaments();
}
