package com.example.data_service.entity;

import com.example.data_service.dto.PostgresIntegerArrayConverter;
import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.util.List;

@Entity
@Table(name = "esports_tournament_placements")
public class TournamentPlacement {
    @Id
    private Integer id;

    @Column(name = "tournament_id")
    private Integer tournamentId;

    private String placement;

    @Column(name = "esports_team_id")
    private Integer esportsTeamId;

    @Column(name = "prize_money")
    private String prizeMoney;

    private String stage;

    @Convert(converter = PostgresIntegerArrayConverter.class)
    private List<Integer> players;

    @jakarta.persistence.ManyToOne
    @jakarta.persistence.JoinColumn(name = "esports_team_id", referencedColumnName = "id", insertable = false, updatable = false)
    private Team teamEntity;

    @jakarta.persistence.ManyToOne
    @jakarta.persistence.JoinColumn(name = "tournament_id", referencedColumnName = "id", insertable = false, updatable = false)
    private Tournament tournamentEntity;

    // getters/setters
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
    public List<Integer> getPlayers() { return players; }
    public void setPlayers(List<Integer> players) { this.players = players; }

    public Team getTeamEntity() { return teamEntity; }
    public Tournament getTournamentEntity() { return tournamentEntity; }
}
