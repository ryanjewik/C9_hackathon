package com.example.data_service.dto;

public class MatchSummaryDto {
    private Integer team1Id;
    private String team1Name;
    private Integer team2Id;
    private String team2Name;

    public MatchSummaryDto() {}

    public MatchSummaryDto(Integer team1Id, String team1Name, Integer team2Id, String team2Name) {
        this.team1Id = team1Id;
        this.team1Name = team1Name;
        this.team2Id = team2Id;
        this.team2Name = team2Name;
    }

    public Integer getTeam1Id() { return team1Id; }
    public void setTeam1Id(Integer team1Id) { this.team1Id = team1Id; }
    public String getTeam1Name() { return team1Name; }
    public void setTeam1Name(String team1Name) { this.team1Name = team1Name; }
    public Integer getTeam2Id() { return team2Id; }
    public void setTeam2Id(Integer team2Id) { this.team2Id = team2Id; }
    public String getTeam2Name() { return team2Name; }
    public void setTeam2Name(String team2Name) { this.team2Name = team2Name; }
}
