package com.example.data_service.dto;

import java.math.BigDecimal;

public class PlayerGameDto {
    private Integer id;
    private Integer matchId;
    private Integer gameId;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer playerId;
    @com.fasterxml.jackson.annotation.JsonProperty("player")
    private PlayerDto playerObj;
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

    public PlayerDto getPlayerObj() { return playerObj; }
    public void setPlayerObj(PlayerDto playerObj) { this.playerObj = playerObj; }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public Integer getMatchId() {
        return matchId;
    }

    public void setMatchId(Integer matchId) {
        this.matchId = matchId;
    }

    public Integer getGameId() {
        return gameId;
    }

    public void setGameId(Integer gameId) {
        this.gameId = gameId;
    }

    public Integer getPlayerId() {
        return playerId;
    }

    public void setPlayerId(Integer playerId) {
        this.playerId = playerId;
    }

    public Integer getTeamId() {
        return teamId;
    }

    public void setTeamId(Integer teamId) {
        this.teamId = teamId;
    }

    public Integer getRosterId() {
        return rosterId;
    }

    public void setRosterId(Integer rosterId) {
        this.rosterId = rosterId;
    }

    public Integer getTournamentId() {
        return tournamentId;
    }

    public void setTournamentId(Integer tournamentId) {
        this.tournamentId = tournamentId;
    }

    public String getMap() {
        return map;
    }

    public void setMap(String map) {
        this.map = map;
    }

    public String getAgent() {
        return agent;
    }

    public void setAgent(String agent) {
        this.agent = agent;
    }

    public java.math.BigDecimal getRating() {
        return rating;
    }

    public void setRating(java.math.BigDecimal rating) {
        this.rating = rating;
    }

    public Integer getAcs() {
        return acs;
    }

    public void setAcs(Integer acs) {
        this.acs = acs;
    }

    public Integer getKills() {
        return kills;
    }

    public void setKills(Integer kills) {
        this.kills = kills;
    }

    public Integer getDeaths() {
        return deaths;
    }

    public void setDeaths(Integer deaths) {
        this.deaths = deaths;
    }

    public Integer getAssists() {
        return assists;
    }

    public void setAssists(Integer assists) {
        this.assists = assists;
    }

    public String getKast() {
        return kast;
    }

    public void setKast(String kast) {
        this.kast = kast;
    }

    public Integer getAdr() {
        return adr;
    }

    public void setAdr(Integer adr) {
        this.adr = adr;
    }

    public String getHsPercent() {
        return hsPercent;
    }

    public void setHsPercent(String hsPercent) {
        this.hsPercent = hsPercent;
    }

    public Integer getFk() {
        return fk;
    }

    public void setFk(Integer fk) {
        this.fk = fk;
    }

    public Integer getFd() {
        return fd;
    }

    public void setFd(Integer fd) {
        this.fd = fd;
    }
}
