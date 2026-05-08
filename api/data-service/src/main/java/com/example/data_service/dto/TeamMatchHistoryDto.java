package com.example.data_service.dto;

import java.time.LocalDateTime;

public class TeamMatchHistoryDto {
    private LocalDateTime date;
    private Boolean won;
    private String opponentName;
    private Integer teamScore;
    private Integer opponentScore;
    private String tournamentName;

    public TeamMatchHistoryDto(LocalDateTime date, Boolean won, String opponentName,
                               Integer teamScore, Integer opponentScore, String tournamentName) {
        this.date = date;
        this.won = won;
        this.opponentName = opponentName;
        this.teamScore = teamScore;
        this.opponentScore = opponentScore;
        this.tournamentName = tournamentName;
    }

    public LocalDateTime getDate() { return date; }
    public void setDate(LocalDateTime date) { this.date = date; }
    public Boolean getWon() { return won; }
    public void setWon(Boolean won) { this.won = won; }
    public String getOpponentName() { return opponentName; }
    public void setOpponentName(String opponentName) { this.opponentName = opponentName; }
    public Integer getTeamScore() { return teamScore; }
    public void setTeamScore(Integer teamScore) { this.teamScore = teamScore; }
    public Integer getOpponentScore() { return opponentScore; }
    public void setOpponentScore(Integer opponentScore) { this.opponentScore = opponentScore; }
    public String getTournamentName() { return tournamentName; }
    public void setTournamentName(String tournamentName) { this.tournamentName = tournamentName; }
}
