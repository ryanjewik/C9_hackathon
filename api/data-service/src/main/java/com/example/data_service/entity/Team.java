package com.example.data_service.entity;

import com.example.data_service.dto.PostgresIntegerArrayConverter;
import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.util.List;

@Entity
@Table(name = "esports_teams")
public class Team {
    @Id
    private Integer id;

    private String name;

    @Column(name = "team_tag")
    private String teamTag;

    private String location;

    @Convert(converter = PostgresIntegerArrayConverter.class)
    private List<Integer> titles;

    @Column(name = "match_wins")
    private Integer matchWins;

    @Column(name = "match_losses")
    private Integer matchLosses;

    @Column(name = "current_roster_id")
    private Integer currentRosterId;

    // getters/setters
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getTeamTag() { return teamTag; }
    public void setTeamTag(String teamTag) { this.teamTag = teamTag; }
    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }
    public List<Integer> getTitles() { return titles; }
    public void setTitles(List<Integer> titles) { this.titles = titles; }
    public Integer getMatchWins() { return matchWins; }
    public void setMatchWins(Integer matchWins) { this.matchWins = matchWins; }
    public Integer getMatchLosses() { return matchLosses; }
    public void setMatchLosses(Integer matchLosses) { this.matchLosses = matchLosses; }
    public Integer getCurrentRosterId() { return currentRosterId; }
    public void setCurrentRosterId(Integer currentRosterId) { this.currentRosterId = currentRosterId; }
}
