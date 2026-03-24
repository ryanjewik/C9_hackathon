package com.example.data_service.dto;

import java.util.List;

public class RosterDto {
    private Integer id;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer teamId;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer player1;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer player2;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer player3;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer player4;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer player5;
    private String dateCreated;
    private Integer mapWins;
    private Integer mapLosses;

    @com.fasterxml.jackson.annotation.JsonProperty("team")
    private TeamDto teamObj;
    @com.fasterxml.jackson.annotation.JsonProperty("player1")
    private PlayerDto player1Obj;
    @com.fasterxml.jackson.annotation.JsonProperty("player2")
    private PlayerDto player2Obj;
    @com.fasterxml.jackson.annotation.JsonProperty("player3")
    private PlayerDto player3Obj;
    @com.fasterxml.jackson.annotation.JsonProperty("player4")
    private PlayerDto player4Obj;
    @com.fasterxml.jackson.annotation.JsonProperty("player5")
    private PlayerDto player5Obj;

    public RosterDto() {}

    // Existing constructor (keeps backward compatibility)
    public RosterDto(Integer id, Integer teamId, Integer player1, Integer player2, Integer player3, Integer player4, Integer player5, String dateCreated, Integer mapWins, Integer mapLosses) {
        this.id = id;
        this.teamId = teamId;
        this.player1 = player1;
        this.player2 = player2;
        this.player3 = player3;
        this.player4 = player4;
        this.player5 = player5;
        this.dateCreated = dateCreated;
        this.mapWins = mapWins;
        this.mapLosses = mapLosses;
    }

    // New constructor used for mapping without players list
    public RosterDto(Integer id, String dateCreated, Integer mapWins, Integer mapLosses) {
        this.id = id;
        this.dateCreated = dateCreated;
        this.mapWins = mapWins;
        this.mapLosses = mapLosses;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public Integer getTeamId() { return teamId; }
    public void setTeamId(Integer teamId) { this.teamId = teamId; }
    public Integer getPlayer1() { return player1; }
    public void setPlayer1(Integer player1) { this.player1 = player1; }
    public Integer getPlayer2() { return player2; }
    public void setPlayer2(Integer player2) { this.player2 = player2; }
    public Integer getPlayer3() { return player3; }
    public void setPlayer3(Integer player3) { this.player3 = player3; }
    public Integer getPlayer4() { return player4; }
    public void setPlayer4(Integer player4) { this.player4 = player4; }
    public Integer getPlayer5() { return player5; }
    public void setPlayer5(Integer player5) { this.player5 = player5; }
    public TeamDto getTeamObj() {return teamObj; }
    public void setTeamObj(TeamDto teamObj) {this.teamObj = teamObj; }
    public PlayerDto getPlayer1Obj() { return player1Obj; }
    public void setPlayer1Obj(PlayerDto player1Obj) { this.player1Obj = player1Obj; }
    public PlayerDto getPlayer2Obj() { return player2Obj; }
    public void setPlayer2Obj(PlayerDto player2Obj) { this.player2Obj = player2Obj; }
    public PlayerDto getPlayer3Obj() { return player3Obj; }
    public void setPlayer3Obj(PlayerDto player3Obj) { this.player3Obj = player3Obj; }
    public PlayerDto getPlayer4Obj() { return player4Obj; }
    public void setPlayer4Obj(PlayerDto player4Obj) { this.player4Obj = player4Obj; }
    public PlayerDto getPlayer5Obj() { return player5Obj; }
    public void setPlayer5Obj(PlayerDto player5Obj) { this.player5Obj = player5Obj; }
    public String getDateCreated() { return dateCreated; }
    public void setDateCreated(String dateCreated) { this.dateCreated = dateCreated; }
    public Integer getMapWins() { return mapWins; }
    public void setMapWins(Integer mapWins) { this.mapWins = mapWins; }
    public Integer getMapLosses() { return mapLosses; }
    public void setMapLosses(Integer mapLosses) { this.mapLosses = mapLosses; }
    
}
