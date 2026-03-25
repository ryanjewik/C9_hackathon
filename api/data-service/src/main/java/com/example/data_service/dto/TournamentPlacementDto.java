package com.example.data_service.dto;

public class TournamentPlacementDto {
    private Integer id;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer tournamentId;
    private String placement;
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Integer esportsTeamId;
    private String prizeMoney;
    private String stage;

    @com.fasterxml.jackson.annotation.JsonProperty("tournament")
    private TournamentDto tournamentObj;

    @com.fasterxml.jackson.annotation.JsonProperty("team")
    private TeamDto teamObj;

    @com.fasterxml.jackson.annotation.JsonProperty("players")
    private java.util.List<PlayerDto> playersObj;

    public TournamentPlacementDto() {}

    public TournamentPlacementDto(Integer id, Integer tournamentId, String placement, Integer esportsTeamId, String prizeMoney, String stage, java.util.List<Integer> players) {
        this.id = id;
        this.tournamentId = tournamentId;
        this.placement = placement;
        this.esportsTeamId = esportsTeamId;
        this.prizeMoney = prizeMoney;
        this.stage = stage;
        // keep legacy constructor behavior for internal use
        if (players != null) this.playersObj = new java.util.ArrayList<>();
    }

    // New constructor used by service mapping
    public TournamentPlacementDto(Integer id, String placement, String prizeMoney, String stage) {
        this.id = id;
        this.placement = placement;
        this.prizeMoney = prizeMoney;
        this.stage = stage;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public Integer getTournamentId() { return tournamentId; }
    public void setTournamentId(Integer tournamentId) { this.tournamentId = tournamentId; }
    public String getPlacement() { return placement; }
    public void setPlacement(String placement) { this.placement = placement; }
    public Integer getEsportsTeamId() { return esportsTeamId; }
    public void setEsportsTeamId(Integer esportsTeamId) { this.esportsTeamId = esportsTeamId; }
    public String getPrizeMoney() { return prizeMoney; }
    public void setPrizeMoney(String prizeMoney) { this.prizeMoney = prizeMoney; }
    public String getStage() { return stage; }
    public void setStage(String stage) { this.stage = stage; }
    public java.util.List<PlayerDto> getPlayersObj() { return playersObj; }
    public void setPlayersObj(java.util.List<PlayerDto> playersObj) { this.playersObj = playersObj; }

    public TournamentDto getTournamentObj() { return tournamentObj; }
    public void setTournamentObj(TournamentDto tournamentObj) { this.tournamentObj = tournamentObj; }

    public TeamDto getTeamObj() { return teamObj; }
    public void setTeamObj(TeamDto teamObj) { this.teamObj = teamObj; }
}
