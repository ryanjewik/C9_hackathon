package com.example.data_service.dto;

public class AgentPickRateDto {
    private String tournamentName;
    private String agent;
    private Long agentPicks;
    private Long totalMatches;
    private Double pickRate;

    public AgentPickRateDto(String tournamentName, String agent, Long agentPicks, Long totalMatches, Double pickRate) {
        this.tournamentName = tournamentName;
        this.agent = agent;
        this.agentPicks = agentPicks;
        this.totalMatches = totalMatches;
        this.pickRate = pickRate;
    }

    public String getTournamentName() { return tournamentName; }
    public void setTournamentName(String tournamentName) { this.tournamentName = tournamentName; }
    public String getAgent() { return agent; }
    public void setAgent(String agent) { this.agent = agent; }
    public Long getAgentPicks() { return agentPicks; }
    public void setAgentPicks(Long agentPicks) { this.agentPicks = agentPicks; }
    public Long getTotalMatches() { return totalMatches; }
    public void setTotalMatches(Long totalMatches) { this.totalMatches = totalMatches; }
    public Double getPickRate() { return pickRate; }
    public void setPickRate(Double pickRate) { this.pickRate = pickRate; }
}
