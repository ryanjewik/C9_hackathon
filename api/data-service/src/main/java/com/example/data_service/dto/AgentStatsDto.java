package com.example.data_service.dto;

public class AgentStatsDto {
    private String nickname;
    private String agent;
    private Double averageRating;
    private Long kills;
    private Long deaths;
    private Long assists;
    private Long firstKills;
    private Long firstDeaths;

    public AgentStatsDto(String nickname, String agent, Double averageRating,
                         Long kills, Long deaths, Long assists, Long firstKills, Long firstDeaths) {
        this.nickname = nickname;
        this.agent = agent;
        this.averageRating = averageRating;
        this.kills = kills;
        this.deaths = deaths;
        this.assists = assists;
        this.firstKills = firstKills;
        this.firstDeaths = firstDeaths;
    }

    public String getNickname() { return nickname; }
    public void setNickname(String nickname) { this.nickname = nickname; }
    public String getAgent() { return agent; }
    public void setAgent(String agent) { this.agent = agent; }
    public Double getAverageRating() { return averageRating; }
    public void setAverageRating(Double averageRating) { this.averageRating = averageRating; }
    public Long getKills() { return kills; }
    public void setKills(Long kills) { this.kills = kills; }
    public Long getDeaths() { return deaths; }
    public void setDeaths(Long deaths) { this.deaths = deaths; }
    public Long getAssists() { return assists; }
    public void setAssists(Long assists) { this.assists = assists; }
    public Long getFirstKills() { return firstKills; }
    public void setFirstKills(Long firstKills) { this.firstKills = firstKills; }
    public Long getFirstDeaths() { return firstDeaths; }
    public void setFirstDeaths(Long firstDeaths) { this.firstDeaths = firstDeaths; }
}
