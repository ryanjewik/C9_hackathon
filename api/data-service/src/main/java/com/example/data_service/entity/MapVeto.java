package com.example.data_service.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "esports_map_veto")
public class MapVeto {
    @Id
    private Integer id;

    @Column(name = "match_id")
    private Integer matchId;

    private String type;

    @Column(name = "team_id")
    private Integer teamId;

    @Column(name = "map_selected")
    private String mapSelected;

    private Integer turn;

    // getters/setters
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
}
