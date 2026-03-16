package com.example.data_service.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.OffsetDateTime;

@Entity
@Table(name = "esports_matches")
public class Match {
    @Id
    private Integer id;

    private String phase;

    private OffsetDateTime date;

    private String patch;

    @Column(name = "tournament_id")
    private Integer tournamentId;

    @Column(name = "tournament_name")
    private String tournamentName;

    @Column(name = "team1_name")
    private String team1Name;

    @Column(name = "team1_id")
    private Integer team1Id;

    @Column(name = "team1_score")
    private Integer team1Score;

    @Column(name = "team2_name")
    private String team2Name;

    @Column(name = "team2_id")
    private Integer team2Id;

    @Column(name = "team2_score")
    private Integer team2Score;

    private Integer winner;
    private String format;
    private String map1;
    private String map2;
    private String map3;
    private String map4;
    private String map5;

    // getters/setters
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public String getPhase() { return phase; }
    public void setPhase(String phase) { this.phase = phase; }
    public OffsetDateTime getDate() { return date; }
    public void setDate(OffsetDateTime date) { this.date = date; }
    public String getPatch() { return patch; }
    public void setPatch(String patch) { this.patch = patch; }
    public Integer getTournamentId() { return tournamentId; }
    public void setTournamentId(Integer tournamentId) { this.tournamentId = tournamentId; }
    public String getTournamentName() { return tournamentName; }
    public void setTournamentName(String tournamentName) { this.tournamentName = tournamentName; }
    public String getTeam1Name() { return team1Name; }
    public void setTeam1Name(String team1Name) { this.team1Name = team1Name; }
    public Integer getTeam1Id() { return team1Id; }
    public void setTeam1Id(Integer team1Id) { this.team1Id = team1Id; }
    public Integer getTeam1Score() { return team1Score; }
    public void setTeam1Score(Integer team1Score) { this.team1Score = team1Score; }
    public String getTeam2Name() { return team2Name; }
    public void setTeam2Name(String team2Name) { this.team2Name = team2Name; }
    public Integer getTeam2Id() { return team2Id; }
    public void setTeam2Id(Integer team2Id) { this.team2Id = team2Id; }
    public Integer getTeam2Score() { return team2Score; }
    public void setTeam2Score(Integer team2Score) { this.team2Score = team2Score; }
    public Integer getWinner() { return winner; }
    public void setWinner(Integer winner) { this.winner = winner; }
    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }
    public String getMap1() { return map1; }
    public void setMap1(String map1) { this.map1 = map1; }
    public String getMap2() { return map2; }
    public void setMap2(String map2) { this.map2 = map2; }
    public String getMap3() { return map3; }
    public void setMap3(String map3) { this.map3 = map3; }
    public String getMap4() { return map4; }
    public void setMap4(String map4) { this.map4 = map4; }
    public String getMap5() { return map5; }
    public void setMap5(String map5) { this.map5 = map5; }
}
