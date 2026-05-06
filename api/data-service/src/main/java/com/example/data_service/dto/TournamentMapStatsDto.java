package com.example.data_service.dto;

public class TournamentMapStatsDto {
    private String tournamentName;
    private String map;
    private Long count;

    public TournamentMapStatsDto(String tournamentName, String map, Long count) {
        this.tournamentName = tournamentName;
        this.map = map;
        this.count = count;
    }

    public String getTournamentName() { return tournamentName; }
    public void setTournamentName(String tournamentName) { this.tournamentName = tournamentName; }
    public String getMap() { return map; }
    public void setMap(String map) { this.map = map; }
    public Long getCount() { return count; }
    public void setCount(Long count) { this.count = count; }
}
