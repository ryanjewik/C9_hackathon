package com.example.data_service.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "esports_game_scores")
public class GameScore {
    @Id
    private Integer id;

    @Column(name = "match_id")
    private Integer matchId;

    @Column(name = "team1_score")
    private Integer team1Score;

    @Column(name = "team2_score")
    private Integer team2Score;

    @Column(name = "team1_id")
    private Integer team1Id;

    @Column(name = "team2_id")
    private Integer team2Id;

    @Column(name = "team1_name")
    private String team1Name;

    @Column(name = "team2_name")
    private String team2Name;

    private String map;
    private Integer winner;

    // getters/setters
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public Integer getMatchId() { return matchId; }
    public void setMatchId(Integer matchId) { this.matchId = matchId; }
    public Integer getTeam1Score() { return team1Score; }
    public void setTeam1Score(Integer team1Score) { this.team1Score = team1Score; }
    public Integer getTeam2Score() { return team2Score; }
    public void setTeam2Score(Integer team2Score) { this.team2Score = team2Score; }
    public Integer getTeam1Id() { return team1Id; }
    public void setTeam1Id(Integer team1Id) { this.team1Id = team1Id; }
    public Integer getTeam2Id() { return team2Id; }
    public void setTeam2Id(Integer team2Id) { this.team2Id = team2Id; }
    public String getTeam1Name() { return team1Name; }
    public void setTeam1Name(String team1Name) { this.team1Name = team1Name; }
    public String getTeam2Name() { return team2Name; }
    public void setTeam2Name(String team2Name) { this.team2Name = team2Name; }
    public String getMap() { return map; }
    public void setMap(String map) { this.map = map; }
    public Integer getWinner() { return winner; }
    public void setWinner(Integer winner) { this.winner = winner; }
}
