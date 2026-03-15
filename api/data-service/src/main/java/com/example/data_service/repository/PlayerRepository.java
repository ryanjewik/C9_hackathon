package com.example.data_service.repository;

import com.example.data_service.entity.Player;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PlayerRepository extends JpaRepository<Player, Integer> {
    //typically the postgres query would be here but the JPA repository will generate the query for us based on the method name and parameters
}