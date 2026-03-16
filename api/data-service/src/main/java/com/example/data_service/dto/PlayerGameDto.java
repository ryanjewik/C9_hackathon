package com.example.data_service.dto;

import java.math.BigDecimal;

public class PlayerGameDto {
    private Integer id;
    private Integer matchId;
    private Integer gameId;
    private Integer playerId;
    private Integer teamId;
    private Integer rosterId;
    private Integer tournamentId;
    private String map;
    private String agent;
    private BigDecimal rating;
    private Integer acs;
    private Integer kills;
    private Integer deaths;
    private Integer assists;
    private String kast;
    private Integer adr;
    private String hsPercent;
    private Integer fk;
    private Integer fd;

    public PlayerGameDto() {}

    public PlayerGameDto(Integer id, Integer matchId, Integer gameId, Integer playerId, Integer teamId, Integer rosterId, Integer tournamentId, String map, String agent, BigDecimal rating, Integer acs, Integer kills, Integer deaths, Integer assists, String kast, Integer adr, String hsPercent, Integer fk, Integer fd) {
        this.id = id;
        this.matchId = matchId;
        this.gameId = gameId;
        this.playerId = playerId;
        this.teamId = teamId;
        this.rosterId = rosterId;
        this.tournamentId = tournamentId;
        this.map = map;
        this.agent = agent;
        this.rating = rating;
        this.acs = acs;
        this.kills = kills;
        this.deaths = deaths;
        this.assists = assists;
        this.kast = kast;
        this.adr = adr;
        this.hsPercent = hsPercent;
        this.fk = fk;
        this.fd = fd;
    }

    // getters and setters omitted for brevity
}
