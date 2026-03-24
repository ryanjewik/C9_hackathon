package com.example.data_service.dto;
import com.fasterxml.jackson.annotation.JsonInclude;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class TeamDto {
    private Integer id;
    private String name;
    private String teamTag;
    private String location;
    private java.util.List<Integer> titles;
    private Integer matchWins;
    private Integer matchLosses;
    private Integer currentRosterId;

    public TeamDto() {}

    public TeamDto(Integer id, String name, String teamTag, String location, java.util.List<Integer> titles, Integer matchWins, Integer matchLosses, Integer currentRosterId) {
        this.id = id;
        this.name = name;
        this.teamTag = teamTag;
        this.location = location;
        this.titles = titles;
        this.matchWins = matchWins;
        this.matchLosses = matchLosses;
        this.currentRosterId = currentRosterId;
    }

    public TeamDto(Integer id, String name, String teamTag) {
        this.id = id;
        this.name = name;
        this.teamTag = teamTag;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getTeamTag() { return teamTag; }
    public void setTeamTag(String teamTag) { this.teamTag = teamTag; }
    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }
    public java.util.List<Integer> getTitles() { return titles; }
    public void setTitles(java.util.List<Integer> titles) { this.titles = titles; }
    public Integer getMatchWins() { return matchWins; }
    public void setMatchWins(Integer matchWins) { this.matchWins = matchWins; }
    public Integer getMatchLosses() { return matchLosses; }
    public void setMatchLosses(Integer matchLosses) { this.matchLosses = matchLosses; }
    public Integer getCurrentRosterId() { return currentRosterId; }
    public void setCurrentRosterId(Integer currentRosterId) { this.currentRosterId = currentRosterId; }
}
