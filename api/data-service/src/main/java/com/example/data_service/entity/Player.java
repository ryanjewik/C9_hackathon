// This patch reflects the current state of the Player.java file without changes.
package com.example.data_service.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.math.BigDecimal;
import java.util.List;
import com.example.data_service.dto.PostgresIntegerArrayConverter;


@Entity
@Table(name = "esports_players")
public class Player {

    @Id
    private Integer id;

    @Column(nullable = false)
    private String nickname;

    @Column(name = "first_name")
    private String firstName;

    @Column(name = "last_name")
    private String lastName;

    private String country;

    @Column(name = "team_id")
    private Integer teamId;

    // Arrays (stored as Postgres integer[]). Convert to JSON lists for API.
    @Column(columnDefinition = "text")
    @Convert(converter = PostgresIntegerArrayConverter.class)
    @JsonProperty("titles")
    private List<Integer> titles;

    // All-time stats
    @Column(name = "all_time_maps")
    @JsonProperty("all_time_maps")
    private Integer allTimeMaps;

    @Column(name = "all_time_map_wins")
    @JsonProperty("all_time_map_wins")
    private Integer allTimeMapWins;

    @Column(name = "all_time_map_losses")
    @JsonProperty("all_time_map_losses")
    private Integer allTimeMapLosses;

    @Column(name = "all_time_rating")
    @JsonProperty("all_time_rating")
    private BigDecimal allTimeRating;

    @Column(name = "all_time_acs")
    @JsonProperty("all_time_acs")
    private BigDecimal allTimeAcs;

    @Column(name = "all_time_kills")
    @JsonProperty("all_time_kills")
    private Integer allTimeKills;

    @Column(name = "all_time_deaths")
    @JsonProperty("all_time_deaths")
    private Integer allTimeDeaths;

    @Column(name = "all_time_assists")
    @JsonProperty("all_time_assists")
    private Integer allTimeAssists;

    @Column(name = "all_time_avg_kills")
    @JsonProperty("all_time_avg_kills")
    private BigDecimal allTimeAvgKills;

    @Column(name = "all_time_avg_deaths")
    @JsonProperty("all_time_avg_deaths")
    private BigDecimal allTimeAvgDeaths;

    @Column(name = "all_time_avg_assists")
    @JsonProperty("all_time_avg_assists")
    private BigDecimal allTimeAvgAssists;

    @Column(name = "all_time_kast")
    @JsonProperty("all_time_kast")
    private BigDecimal allTimeKast;

    @Column(name = "all_time_adr")
    @JsonProperty("all_time_adr")
    private BigDecimal allTimeAdr;

    @Column(name = "all_time_hs_percent")
    @JsonProperty("all_time_hs_percent")
    private BigDecimal allTimeHsPercent;

    @Column(name = "all_time_fk")
    @JsonProperty("all_time_fk")
    private Integer allTimeFk;

    @Column(name = "all_time_fd")
    @JsonProperty("all_time_fd")
    private Integer allTimeFd;

    @Column(name = "all_time_avg_fk")
    @JsonProperty("all_time_avg_fk")
    private BigDecimal allTimeAvgFk;

    @Column(name = "all_time_avg_fd")
    @JsonProperty("all_time_avg_fd")
    private BigDecimal allTimeAvgFd;

    // Last 60 days stats
    @Column(name = "last_60_maps")
    @JsonProperty("last_60_maps")
    private Integer last60Maps;

    @Column(name = "last_60_map_wins")
    @JsonProperty("last_60_map_wins")
    private Integer last60MapWins;

    @Column(name = "last_60_map_losses")
    @JsonProperty("last_60_map_losses")
    private Integer last60MapLosses;

    @Column(name = "last_60_rating")
    @JsonProperty("last_60_rating")
    private BigDecimal last60Rating;

    @Column(name = "last_60_acs")
    @JsonProperty("last_60_acs")
    private BigDecimal last60Acs;

    @Column(name = "last_60_kills")
    @JsonProperty("last_60_kills")
    private Integer last60Kills;

    @Column(name = "last_60_deaths")
    @JsonProperty("last_60_deaths")
    private Integer last60Deaths;

    @Column(name = "last_60_assists")
    @JsonProperty("last_60_assists")
    private Integer last60Assists;

    @Column(name = "last_60_avg_kills")
    @JsonProperty("last_60_avg_kills")
    private BigDecimal last60AvgKills;

    @Column(name = "last_60_avg_deaths")
    @JsonProperty("last_60_avg_deaths")
    private BigDecimal last60AvgDeaths;

    @Column(name = "last_60_avg_assists")
    @JsonProperty("last_60_avg_assists")
    private BigDecimal last60AvgAssists;

    @Column(name = "last_60_kast")
    @JsonProperty("last_60_kast")
    private BigDecimal last60Kast;

    @Column(name = "last_60_adr")
    @JsonProperty("last_60_adr")
    private BigDecimal last60Adr;

    @Column(name = "last_60_hs_percent")
    @JsonProperty("last_60_hs_percent")
    private BigDecimal last60HsPercent;

    @Column(name = "last_60_fk")
    @JsonProperty("last_60_fk")
    private Integer last60Fk;

    @Column(name = "last_60_fd")
    @JsonProperty("last_60_fd")
    private Integer last60Fd;

    @Column(name = "last_60_avg_fk")
    @JsonProperty("last_60_avg_fk")
    private BigDecimal last60AvgFk;

    @Column(name = "last_60_avg_fd")
    @JsonProperty("last_60_avg_fd")
    private BigDecimal last60AvgFd;

    public Player() {}

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }

    public String getNickname() { return nickname; }
    public void setNickname(String nickname) { this.nickname = nickname; }

    public String getFirstName() { return firstName; }
    public void setFirstName(String firstName) { this.firstName = firstName; }

    public String getLastName() { return lastName; }
    public void setLastName(String lastName) { this.lastName = lastName; }

    public String getCountry() { return country; }
    public void setCountry(String country) { this.country = country; }

    public Integer getTeamId() { return teamId; }
    public void setTeamId(Integer teamId) { this.teamId = teamId; }
    public List<Integer> getTitles() { return titles; }
    public void setTitles(List<Integer> titles) { this.titles = titles; }

    public Integer getAllTimeMaps() { return allTimeMaps; }
    public void setAllTimeMaps(Integer allTimeMaps) { this.allTimeMaps = allTimeMaps; }

    public Integer getAllTimeMapWins() { return allTimeMapWins; }
    public void setAllTimeMapWins(Integer allTimeMapWins) { this.allTimeMapWins = allTimeMapWins; }

    public Integer getAllTimeMapLosses() { return allTimeMapLosses; }
    public void setAllTimeMapLosses(Integer allTimeMapLosses) { this.allTimeMapLosses = allTimeMapLosses; }

    public BigDecimal getAllTimeRating() { return allTimeRating; }
    public void setAllTimeRating(BigDecimal allTimeRating) { this.allTimeRating = allTimeRating; }

    public BigDecimal getAllTimeAcs() { return allTimeAcs; }
    public void setAllTimeAcs(BigDecimal allTimeAcs) { this.allTimeAcs = allTimeAcs; }

    public Integer getAllTimeKills() { return allTimeKills; }
    public void setAllTimeKills(Integer allTimeKills) { this.allTimeKills = allTimeKills; }

    public Integer getAllTimeDeaths() { return allTimeDeaths; }
    public void setAllTimeDeaths(Integer allTimeDeaths) { this.allTimeDeaths = allTimeDeaths; }

    public Integer getAllTimeAssists() { return allTimeAssists; }
    public void setAllTimeAssists(Integer allTimeAssists) { this.allTimeAssists = allTimeAssists; }

    public BigDecimal getAllTimeAvgKills() { return allTimeAvgKills; }
    public void setAllTimeAvgKills(BigDecimal allTimeAvgKills) { this.allTimeAvgKills = allTimeAvgKills; }

    public BigDecimal getAllTimeAvgDeaths() { return allTimeAvgDeaths; }
    public void setAllTimeAvgDeaths(BigDecimal allTimeAvgDeaths) { this.allTimeAvgDeaths = allTimeAvgDeaths; }

    public BigDecimal getAllTimeAvgAssists() { return allTimeAvgAssists; }
    public void setAllTimeAvgAssists(BigDecimal allTimeAvgAssists) { this.allTimeAvgAssists = allTimeAvgAssists; }

    public BigDecimal getAllTimeKast() { return allTimeKast; }
    public void setAllTimeKast(BigDecimal allTimeKast) { this.allTimeKast = allTimeKast; }

    public BigDecimal getAllTimeAdr() { return allTimeAdr; }
    public void setAllTimeAdr(BigDecimal allTimeAdr) { this.allTimeAdr = allTimeAdr; }

    public BigDecimal getAllTimeHsPercent() { return allTimeHsPercent; }
    public void setAllTimeHsPercent(BigDecimal allTimeHsPercent) { this.allTimeHsPercent = allTimeHsPercent; }

    public Integer getAllTimeFk() { return allTimeFk; }
    public void setAllTimeFk(Integer allTimeFk) { this.allTimeFk = allTimeFk; }

    public Integer getAllTimeFd() { return allTimeFd; }
    public void setAllTimeFd(Integer allTimeFd) { this.allTimeFd = allTimeFd; }

    public BigDecimal getAllTimeAvgFk() { return allTimeAvgFk; }
    public void setAllTimeAvgFk(BigDecimal allTimeAvgFk) { this.allTimeAvgFk = allTimeAvgFk; }

    public BigDecimal getAllTimeAvgFd() { return allTimeAvgFd; }
    public void setAllTimeAvgFd(BigDecimal allTimeAvgFd) { this.allTimeAvgFd = allTimeAvgFd; }

    public Integer getLast60Maps() { return last60Maps; }
    public void setLast60Maps(Integer last60Maps) { this.last60Maps = last60Maps; }

    public Integer getLast60MapWins() { return last60MapWins; }
    public void setLast60MapWins(Integer last60MapWins) { this.last60MapWins = last60MapWins; }

    public Integer getLast60MapLosses() { return last60MapLosses; }
    public void setLast60MapLosses(Integer last60MapLosses) { this.last60MapLosses = last60MapLosses; }

    public BigDecimal getLast60Rating() { return last60Rating; }
    public void setLast60Rating(BigDecimal last60Rating) { this.last60Rating = last60Rating; }

    public BigDecimal getLast60Acs() { return last60Acs; }
    public void setLast60Acs(BigDecimal last60Acs) { this.last60Acs = last60Acs; }

    public Integer getLast60Kills() { return last60Kills; }
    public void setLast60Kills(Integer last60Kills) { this.last60Kills = last60Kills; }

    public Integer getLast60Deaths() { return last60Deaths; }
    public void setLast60Deaths(Integer last60Deaths) { this.last60Deaths = last60Deaths; }

    public Integer getLast60Assists() { return last60Assists; }
    public void setLast60Assists(Integer last60Assists) { this.last60Assists = last60Assists; }

    public BigDecimal getLast60AvgKills() { return last60AvgKills; }
    public void setLast60AvgKills(BigDecimal last60AvgKills) { this.last60AvgKills = last60AvgKills; }

    public BigDecimal getLast60AvgDeaths() { return last60AvgDeaths; }
    public void setLast60AvgDeaths(BigDecimal last60AvgDeaths) { this.last60AvgDeaths = last60AvgDeaths; }

    public BigDecimal getLast60AvgAssists() { return last60AvgAssists; }
    public void setLast60AvgAssists(BigDecimal last60AvgAssists) { this.last60AvgAssists = last60AvgAssists; }

    public BigDecimal getLast60Kast() { return last60Kast; }
    public void setLast60Kast(BigDecimal last60Kast) { this.last60Kast = last60Kast; }

    public BigDecimal getLast60Adr() { return last60Adr; }
    public void setLast60Adr(BigDecimal last60Adr) { this.last60Adr = last60Adr; }

    public BigDecimal getLast60HsPercent() { return last60HsPercent; }
    public void setLast60HsPercent(BigDecimal last60HsPercent) { this.last60HsPercent = last60HsPercent; }

    public Integer getLast60Fk() { return last60Fk; }
    public void setLast60Fk(Integer last60Fk) { this.last60Fk = last60Fk; }

    public Integer getLast60Fd() { return last60Fd; }
    public void setLast60Fd(Integer last60Fd) { this.last60Fd = last60Fd; }

    public BigDecimal getLast60AvgFk() { return last60AvgFk; }
    public void setLast60AvgFk(BigDecimal last60AvgFk) { this.last60AvgFk = last60AvgFk; }

    public BigDecimal getLast60AvgFd() { return last60AvgFd; }
    public void setLast60AvgFd(BigDecimal last60AvgFd) { this.last60AvgFd = last60AvgFd; }
}