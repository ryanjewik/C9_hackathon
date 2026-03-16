package com.example.data_service.dto;

public class TournamentPlacementDto {
    private Integer id;
    private Integer tournamentId;
    private String placement;
    private Integer esportsTeamId;
    private String prizeMoney;
    private String stage;
    private java.util.List<Integer> players;

    public TournamentPlacementDto() {}

    public TournamentPlacementDto(Integer id, Integer tournamentId, String placement, Integer esportsTeamId, String prizeMoney, String stage, java.util.List<Integer> players) {
        this.id = id;
        this.tournamentId = tournamentId;
        this.placement = placement;
        this.esportsTeamId = esportsTeamId;
        this.prizeMoney = prizeMoney;
        this.stage = stage;
        this.players = players;
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
    public java.util.List<Integer> getPlayers() { return players; }
    public void setPlayers(java.util.List<Integer> players) { this.players = players; }
}
