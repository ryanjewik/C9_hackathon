package com.example.data_service.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.math.BigDecimal;
import java.util.List;

public class PlayerDto {
    private Integer id;
    private String nickname;
    @JsonProperty("first_name")
    private String firstName;
    @JsonProperty("last_name")
    private String lastName;
    private String country;
    @JsonProperty("team_id")
    private Integer teamId;
    private List<Integer> titles;

    @JsonProperty("all_time_maps")
    private Integer allTimeMaps;
    @JsonProperty("all_time_map_wins")
    private Integer allTimeMapWins;
    @JsonProperty("all_time_map_losses")
    private Integer allTimeMapLosses;
    @JsonProperty("all_time_rating")
    private BigDecimal allTimeRating;
    @JsonProperty("all_time_acs")
    private BigDecimal allTimeAcs;
    @JsonProperty("all_time_kills")
    private Integer allTimeKills;
    @JsonProperty("all_time_deaths")
    private Integer allTimeDeaths;
    @JsonProperty("all_time_assists")
    private Integer allTimeAssists;
    @JsonProperty("all_time_avg_kills")
    private BigDecimal allTimeAvgKills;
    @JsonProperty("all_time_avg_deaths")
    private BigDecimal allTimeAvgDeaths;
    @JsonProperty("all_time_avg_assists")
    private BigDecimal allTimeAvgAssists;
    @JsonProperty("all_time_kast")
    private BigDecimal allTimeKast;
    @JsonProperty("all_time_adr")
    private BigDecimal allTimeAdr;
    @JsonProperty("all_time_hs_percent")
    private BigDecimal allTimeHsPercent;
    @JsonProperty("all_time_fk")
    private Integer allTimeFk;
    @JsonProperty("all_time_fd")
    private Integer allTimeFd;
    @JsonProperty("all_time_avg_fk")
    private BigDecimal allTimeAvgFk;
    @JsonProperty("all_time_avg_fd")
    private BigDecimal allTimeAvgFd;

    @JsonProperty("last_60_maps")
    private Integer last60Maps;
    @JsonProperty("last_60_map_wins")
    private Integer last60MapWins;
    @JsonProperty("last_60_map_losses")
    private Integer last60MapLosses;
    @JsonProperty("last_60_rating")
    private BigDecimal last60Rating;
    @JsonProperty("last_60_acs")
    private BigDecimal last60Acs;
    @JsonProperty("last_60_kills")
    private Integer last60Kills;
    @JsonProperty("last_60_deaths")
    private Integer last60Deaths;
    @JsonProperty("last_60_assists")
    private Integer last60Assists;
    @JsonProperty("last_60_avg_kills")
    private BigDecimal last60AvgKills;
    @JsonProperty("last_60_avg_deaths")
    private BigDecimal last60AvgDeaths;
    @JsonProperty("last_60_avg_assists")
    private BigDecimal last60AvgAssists;
    @JsonProperty("last_60_kast")
    private BigDecimal last60Kast;
    @JsonProperty("last_60_adr")
    private BigDecimal last60Adr;
    @JsonProperty("last_60_hs_percent")
    private BigDecimal last60HsPercent;
    @JsonProperty("last_60_fk")
    private Integer last60Fk;
    @JsonProperty("last_60_fd")
    private Integer last60Fd;
    @JsonProperty("last_60_avg_fk")
    private BigDecimal last60AvgFk;
    @JsonProperty("last_60_avg_fd")
    private BigDecimal last60AvgFd;

    public PlayerDto() {}

    // Full-args constructor used for JPA constructor projections
    public PlayerDto(Integer id,
                     String nickname,
                     String firstName,
                     String lastName,
                     String country,
                     Integer teamId,
                     java.util.List<Integer> titles,
                     Integer allTimeMaps,
                     Integer allTimeMapWins,
                     Integer allTimeMapLosses,
                     BigDecimal allTimeRating,
                     BigDecimal allTimeAcs,
                     Integer allTimeKills,
                     Integer allTimeDeaths,
                     Integer allTimeAssists,
                     BigDecimal allTimeAvgKills,
                     BigDecimal allTimeAvgDeaths,
                     BigDecimal allTimeAvgAssists,
                     BigDecimal allTimeKast,
                     BigDecimal allTimeAdr,
                     BigDecimal allTimeHsPercent,
                     Integer allTimeFk,
                     Integer allTimeFd,
                     BigDecimal allTimeAvgFk,
                     BigDecimal allTimeAvgFd,
                     Integer last60Maps,
                     Integer last60MapWins,
                     Integer last60MapLosses,
                     BigDecimal last60Rating,
                     BigDecimal last60Acs,
                     Integer last60Kills,
                     Integer last60Deaths,
                     Integer last60Assists,
                     BigDecimal last60AvgKills,
                     BigDecimal last60AvgDeaths,
                     BigDecimal last60AvgAssists,
                     BigDecimal last60Kast,
                     BigDecimal last60Adr,
                     BigDecimal last60HsPercent,
                     Integer last60Fk,
                     Integer last60Fd,
                     BigDecimal last60AvgFk,
                     BigDecimal last60AvgFd) {
        this.id = id;
        this.nickname = nickname;
        this.firstName = firstName;
        this.lastName = lastName;
        this.country = country;
        this.teamId = teamId;
        this.titles = titles;
        this.allTimeMaps = allTimeMaps;
        this.allTimeMapWins = allTimeMapWins;
        this.allTimeMapLosses = allTimeMapLosses;
        this.allTimeRating = allTimeRating;
        this.allTimeAcs = allTimeAcs;
        this.allTimeKills = allTimeKills;
        this.allTimeDeaths = allTimeDeaths;
        this.allTimeAssists = allTimeAssists;
        this.allTimeAvgKills = allTimeAvgKills;
        this.allTimeAvgDeaths = allTimeAvgDeaths;
        this.allTimeAvgAssists = allTimeAvgAssists;
        this.allTimeKast = allTimeKast;
        this.allTimeAdr = allTimeAdr;
        this.allTimeHsPercent = allTimeHsPercent;
        this.allTimeFk = allTimeFk;
        this.allTimeFd = allTimeFd;
        this.allTimeAvgFk = allTimeAvgFk;
        this.allTimeAvgFd = allTimeAvgFd;
        this.last60Maps = last60Maps;
        this.last60MapWins = last60MapWins;
        this.last60MapLosses = last60MapLosses;
        this.last60Rating = last60Rating;
        this.last60Acs = last60Acs;
        this.last60Kills = last60Kills;
        this.last60Deaths = last60Deaths;
        this.last60Assists = last60Assists;
        this.last60AvgKills = last60AvgKills;
        this.last60AvgDeaths = last60AvgDeaths;
        this.last60AvgAssists = last60AvgAssists;
        this.last60Kast = last60Kast;
        this.last60Adr = last60Adr;
        this.last60HsPercent = last60HsPercent;
        this.last60Fk = last60Fk;
        this.last60Fd = last60Fd;
        this.last60AvgFk = last60AvgFk;
        this.last60AvgFd = last60AvgFd;
    }

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

    public java.util.List<Integer> getTitles() { return titles; }
    public void setTitles(java.util.List<Integer> titles) { this.titles = titles; }

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
