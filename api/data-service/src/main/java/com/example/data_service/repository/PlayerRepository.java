package com.example.data_service.repository;

import com.example.data_service.dto.PlayerDto;
import com.example.data_service.entity.Player;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface PlayerRepository extends JpaRepository<Player, Integer> {
    // Constructor-based projection: fetch PlayerDto directly from JPQL
    @Query("select new com.example.data_service.dto.PlayerDto(p.id, p.nickname, p.firstName, p.lastName, p.country, p.teamId, p.titles, p.allTimeMaps, p.allTimeMapWins, p.allTimeMapLosses, p.allTimeRating, p.allTimeAcs, p.allTimeKills, p.allTimeDeaths, p.allTimeAssists, p.allTimeAvgKills, p.allTimeAvgDeaths, p.allTimeAvgAssists, p.allTimeKast, p.allTimeAdr, p.allTimeHsPercent, p.allTimeFk, p.allTimeFd, p.allTimeAvgFk, p.allTimeAvgFd, p.last60Maps, p.last60MapWins, p.last60MapLosses, p.last60Rating, p.last60Acs, p.last60Kills, p.last60Deaths, p.last60Assists, p.last60AvgKills, p.last60AvgDeaths, p.last60AvgAssists, p.last60Kast, p.last60Adr, p.last60HsPercent, p.last60Fk, p.last60Fd, p.last60AvgFk, p.last60AvgFd) from Player p")
    Page<PlayerDto> findAllAsDto(Pageable pageable);

    @Query("select new com.example.data_service.dto.PlayerDto(p.id, p.nickname, p.firstName, p.lastName, p.country, p.teamId, p.titles, p.allTimeMaps, p.allTimeMapWins, p.allTimeMapLosses, p.allTimeRating, p.allTimeAcs, p.allTimeKills, p.allTimeDeaths, p.allTimeAssists, p.allTimeAvgKills, p.allTimeAvgDeaths, p.allTimeAvgAssists, p.allTimeKast, p.allTimeAdr, p.allTimeHsPercent, p.allTimeFk, p.allTimeFd, p.allTimeAvgFk, p.allTimeAvgFd, p.last60Maps, p.last60MapWins, p.last60MapLosses, p.last60Rating, p.last60Acs, p.last60Kills, p.last60Deaths, p.last60Assists, p.last60AvgKills, p.last60AvgDeaths, p.last60AvgAssists, p.last60Kast, p.last60Adr, p.last60HsPercent, p.last60Fk, p.last60Fd, p.last60AvgFk, p.last60AvgFd) from Player p where p.id = :id")
    PlayerDto findDtoById(@Param("id") Integer id);
}