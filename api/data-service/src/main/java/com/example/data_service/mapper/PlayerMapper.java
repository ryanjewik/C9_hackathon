package com.example.data_service.mapper;

import com.example.data_service.dto.PlayerDto;
import com.example.data_service.entity.Player;

public class PlayerMapper {
    public static PlayerDto toDto(Player p) {
        if (p == null) return null;
        PlayerDto d = new PlayerDto();
        d.setId(p.getId());
        d.setNickname(p.getNickname());
        d.setFirstName(p.getFirstName());
        d.setLastName(p.getLastName());
        d.setCountry(p.getCountry());
        d.setTeamId(p.getTeamId());
        d.setTitles(p.getTitles());

        d.setAllTimeMaps(p.getAllTimeMaps());
        d.setAllTimeMapWins(p.getAllTimeMapWins());
        d.setAllTimeMapLosses(p.getAllTimeMapLosses());
        d.setAllTimeRating(p.getAllTimeRating());
        d.setAllTimeAcs(p.getAllTimeAcs());
        d.setAllTimeKills(p.getAllTimeKills());
        d.setAllTimeDeaths(p.getAllTimeDeaths());
        d.setAllTimeAssists(p.getAllTimeAssists());
        d.setAllTimeAvgKills(p.getAllTimeAvgKills());
        d.setAllTimeAvgDeaths(p.getAllTimeAvgDeaths());
        d.setAllTimeAvgAssists(p.getAllTimeAvgAssists());
        d.setAllTimeKast(p.getAllTimeKast());
        d.setAllTimeAdr(p.getAllTimeAdr());
        d.setAllTimeHsPercent(p.getAllTimeHsPercent());
        d.setAllTimeFk(p.getAllTimeFk());
        d.setAllTimeFd(p.getAllTimeFd());
        d.setAllTimeAvgFk(p.getAllTimeAvgFk());
        d.setAllTimeAvgFd(p.getAllTimeAvgFd());

        d.setLast60Maps(p.getLast60Maps());
        d.setLast60MapWins(p.getLast60MapWins());
        d.setLast60MapLosses(p.getLast60MapLosses());
        d.setLast60Rating(p.getLast60Rating());
        d.setLast60Acs(p.getLast60Acs());
        d.setLast60Kills(p.getLast60Kills());
        d.setLast60Deaths(p.getLast60Deaths());
        d.setLast60Assists(p.getLast60Assists());
        d.setLast60AvgKills(p.getLast60AvgKills());
        d.setLast60AvgDeaths(p.getLast60AvgDeaths());
        d.setLast60AvgAssists(p.getLast60AvgAssists());
        d.setLast60Kast(p.getLast60Kast());
        d.setLast60Adr(p.getLast60Adr());
        d.setLast60HsPercent(p.getLast60HsPercent());
        d.setLast60Fk(p.getLast60Fk());
        d.setLast60Fd(p.getLast60Fd());
        d.setLast60AvgFk(p.getLast60AvgFk());
        d.setLast60AvgFd(p.getLast60AvgFd());

        return d;
    }
}
