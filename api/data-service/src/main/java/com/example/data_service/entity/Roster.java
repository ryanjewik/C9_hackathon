package com.example.data_service.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Table;

@Entity
@Table(name = "esports_rosters")
public class Roster {
    @Id
    private Integer id;

    @Column(name = "team_id")
    private Integer teamId;

    @Column(name = "player_1")
    private Integer player1;

    @Column(name = "player_2")
    private Integer player2;

    @Column(name = "player_3")
    private Integer player3;

    @Column(name = "player_4")
    private Integer player4;

    @Column(name = "player_5")
    private Integer player5;

    @ManyToOne
    @JoinColumn(name = "team_id", referencedColumnName = "id", insertable = false, updatable = false)
    private Team teamEntity;

    @ManyToOne
    @JoinColumn(name = "player_1", referencedColumnName = "id", insertable = false, updatable = false)
    private Player player1Entity;

    @ManyToOne
    @JoinColumn(name = "player_2", referencedColumnName = "id", insertable = false, updatable = false)
    private Player player2Entity;

    @ManyToOne
    @JoinColumn(name = "player_3", referencedColumnName = "id", insertable = false, updatable = false)
    private Player player3Entity;

    @ManyToOne
    @JoinColumn(name = "player_4", referencedColumnName = "id", insertable = false, updatable = false)
    private Player player4Entity;

    @ManyToOne
    @JoinColumn(name = "player_5", referencedColumnName = "id", insertable = false, updatable = false)
    private Player player5Entity;

    @Column(name = "date_created")
    private String dateCreated;

    @Column(name = "map_wins")
    private Integer mapWins;

    @Column(name = "map_losses")
    private Integer mapLosses;

    // getters/setters
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
    public String getDateCreated() { return dateCreated; }
    public void setDateCreated(String dateCreated) { this.dateCreated = dateCreated; }
    public Integer getMapWins() { return mapWins; }
    public void setMapWins(Integer mapWins) { this.mapWins = mapWins; }
    public Integer getMapLosses() { return mapLosses; }
    public void setMapLosses(Integer mapLosses) { this.mapLosses = mapLosses; }
    public Player getPlayer1Entity() { return player1Entity; }
    public Player getPlayer2Entity() { return player2Entity; }
    public Player getPlayer3Entity() { return player3Entity; }
    public Player getPlayer4Entity() { return player4Entity; }
    public Player getPlayer5Entity() { return player5Entity; }
    public Team getTeamEntity() { return teamEntity; }
}
