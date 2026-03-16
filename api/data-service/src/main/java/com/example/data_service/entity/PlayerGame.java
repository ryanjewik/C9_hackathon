package com.example.data_service.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.math.BigDecimal;

@Entity
@Table(name = "esports_player_games")
public class PlayerGame {
    @Id
    private Integer id;

    @Column(name = "match_id")
    private Integer matchId;

    @Column(name = "game_id")
    private Integer gameId;

    @Column(name = "player_id")
    private Integer playerId;

    @Column(name = "team_id")
    private Integer teamId;

    @Column(name = "roster_id")
    private Integer rosterId;

    @Column(name = "tournament_id")
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

    @Column(name = "hs_percent")
    private String hsPercent;

    private Integer fk;
    private Integer fd;

    // getters/setters omitted for brevity - include as needed
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public Integer getMatchId() { return matchId; }
    public void setMatchId(Integer matchId) { this.matchId = matchId; }
    public Integer getGameId() { return gameId; }
    public void setGameId(Integer gameId) { this.gameId = gameId; }
    public Integer getPlayerId() { return playerId; }
    public void setPlayerId(Integer playerId) { this.playerId = playerId; }
    public Integer getTeamId() { return teamId; }
    public void setTeamId(Integer teamId) { this.teamId = teamId; }
    public Integer getRosterId() { return rosterId; }
    public void setRosterId(Integer rosterId) { this.rosterId = rosterId; }
    public Integer getTournamentId() { return tournamentId; }
    public void setTournamentId(Integer tournamentId) { this.tournamentId = tournamentId; }
    public String getMap() { return map; }
    public void setMap(String map) { this.map = map; }
    public String getAgent() { return agent; }
    public void setAgent(String agent) { this.agent = agent; }
    public BigDecimal getRating() { return rating; }
    public void setRating(BigDecimal rating) { this.rating = rating; }
    public Integer getAcs() { return acs; }
    public void setAcs(Integer acs) { this.acs = acs; }
    public Integer getKills() { return kills; }
    public void setKills(Integer kills) { this.kills = kills; }
    public Integer getDeaths() { return deaths; }
    public void setDeaths(Integer deaths) { this.deaths = deaths; }
    public Integer getAssists() { return assists; }
    public void setAssists(Integer assists) { this.assists = assists; }
    public String getKast() { return kast; }
    public void setKast(String kast) { this.kast = kast; }
    public Integer getAdr() { return adr; }
    public void setAdr(Integer adr) { this.adr = adr; }
    public String getHsPercent() { return hsPercent; }
    public void setHsPercent(String hsPercent) { this.hsPercent = hsPercent; }
    public Integer getFk() { return fk; }
    public void setFk(Integer fk) { this.fk = fk; }
    public Integer getFd() { return fd; }
    public void setFd(Integer fd) { this.fd = fd; }
}
