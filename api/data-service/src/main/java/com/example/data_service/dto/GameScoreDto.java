package com.example.data_service.dto;

public class GameScoreDto {
    private Integer id;
    private Integer matchId;
    private Integer team1Score;
    private Integer team2Score;
    private Integer team1Id;
    private Integer team2Id;
    private String team1Name;
    private String team2Name;
    private String map;
    private Integer winner;

    public GameScoreDto() {}

    public GameScoreDto(Integer id, Integer matchId, Integer team1Score, Integer team2Score, Integer team1Id, Integer team2Id, String team1Name, String team2Name, String map, Integer winner) {
        this.id = id;
        this.matchId = matchId;
        this.team1Score = team1Score;
        this.team2Score = team2Score;
        this.team1Id = team1Id;
        this.team2Id = team2Id;
        this.team1Name = team1Name;
        this.team2Name = team2Name;
        this.map = map;
        this.winner = winner;
    }

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
