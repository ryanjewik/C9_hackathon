package com.example.data_service.dto;

public class MapVetoDto {
    private Integer id;
    private String type;
    
    private String mapSelected;
    private Integer turn;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer matchId;

    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer teamId;

    @com.fasterxml.jackson.annotation.JsonProperty("team")
    private TeamDto teamObj;
    @com.fasterxml.jackson.annotation.JsonProperty("match")
    private MatchSummaryDto matchObj;

    public MapVetoDto() {}

    public MapVetoDto(Integer id, Integer matchId, String type, Integer teamId, String mapSelected, Integer turn) {
        this.id = id;
        this.matchId = matchId;
        this.type = type;
        this.teamId = teamId;
        this.mapSelected = mapSelected;
        this.turn = turn;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public Integer getMatchId() { return matchId; }
    public void setMatchId(Integer matchId) { this.matchId = matchId; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public Integer getTeamId() { return teamId; }
    public void setTeamId(Integer teamId) { this.teamId = teamId; }
    public String getMapSelected() { return mapSelected; }
    public void setMapSelected(String mapSelected) { this.mapSelected = mapSelected; }
    public Integer getTurn() { return turn; }
    public void setTurn(Integer turn) { this.turn = turn; }
    public TeamDto getTeamObj() { return teamObj; }
    public void setTeamObj(TeamDto teamObj) { this.teamObj = teamObj; }
    public MatchSummaryDto getMatchObj() { return matchObj; }
    public void setMatchObj(MatchSummaryDto matchObj) { this.matchObj = matchObj; }
}
