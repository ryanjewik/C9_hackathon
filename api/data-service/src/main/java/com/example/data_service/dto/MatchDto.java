package com.example.data_service.dto;

public class MatchDto {
    private Integer id;
    private String phase;
    private java.time.OffsetDateTime date;
    private String patch;
    private Integer tournamentId;
    private String tournamentName;
    private String team1Name;
    private Integer team1Id;
    private Integer team1Score;
    private String team2Name;
    private Integer team2Id;
    private Integer team2Score;
    private Integer winner;
    private String format;
    private String map1;
    private String map2;
    private String map3;
    private String map4;
    private String map5;
    private java.util.List<Integer> mapVetoIds;
    private java.util.List<Integer> gameScoreIds;

    public MatchDto() {}

    public MatchDto(Integer id, String phase, java.time.OffsetDateTime date, String patch, Integer tournamentId, String tournamentName, String team1Name, Integer team1Id, Integer team1Score, String team2Name, Integer team2Id, Integer team2Score, Integer winner, String format, String map1, String map2, String map3, String map4, String map5) {
        this.id = id;
        this.phase = phase;
        this.date = date;
        this.patch = patch;
        this.tournamentId = tournamentId;
        this.tournamentName = tournamentName;
        this.team1Name = team1Name;
        this.team1Id = team1Id;
        this.team1Score = team1Score;
        this.team2Name = team2Name;
        this.team2Id = team2Id;
        this.team2Score = team2Score;
        this.winner = winner;
        this.format = format;
        this.map1 = map1;
        this.map2 = map2;
        this.map3 = map3;
        this.map4 = map4;
        this.map5 = map5;
    }

    // getters/setters omitted for brevity (add as needed)
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public String getPhase() { return phase; }
    public void setPhase(String phase) { this.phase = phase; }
    public java.time.OffsetDateTime getDate() { return date; }
    public void setDate(java.time.OffsetDateTime date) { this.date = date; }
    public String getPatch() { return patch; }
    public void setPatch(String patch) { this.patch = patch; }
    public Integer getTournamentId() { return tournamentId; }
    public void setTournamentId(Integer tournamentId) { this.tournamentId = tournamentId; }
    public String getTournamentName() { return tournamentName; }
    public void setTournamentName(String tournamentName) { this.tournamentName = tournamentName; }
    public String getTeam1Name() { return team1Name; }
    public void setTeam1Name(String team1Name) { this.team1Name = team1Name; }
    public Integer getTeam1Id() { return team1Id; }
    public void setTeam1Id(Integer team1Id) { this.team1Id = team1Id; }
    public Integer getTeam1Score() { return team1Score; }
    public void setTeam1Score(Integer team1Score) { this.team1Score = team1Score; }
    public String getTeam2Name() { return team2Name; }
    public void setTeam2Name(String team2Name) { this.team2Name = team2Name; }
    public Integer getTeam2Id() { return team2Id; }
    public void setTeam2Id(Integer team2Id) { this.team2Id = team2Id; }
    public Integer getTeam2Score() { return team2Score; }
    public void setTeam2Score(Integer team2Score) { this.team2Score = team2Score; }
    public Integer getWinner() { return winner; }
    public void setWinner(Integer winner) { this.winner = winner; }
    public String getFormat() { return format; }
    public void setFormat(String format) { this.format = format; }
    public String getMap1() { return map1; }
    public void setMap1(String map1) { this.map1 = map1; }
    public String getMap2() { return map2; }
    public void setMap2(String map2) { this.map2 = map2; }
    public String getMap3() { return map3; }
    public void setMap3(String map3) { this.map3 = map3; }
    public String getMap4() { return map4; }
    public void setMap4(String map4) { this.map4 = map4; }
    public String getMap5() { return map5; }
    public void setMap5(String map5) { this.map5 = map5; }
    public java.util.List<Integer> getMapVetoIds() { return mapVetoIds; }
    public void setMapVetoIds(java.util.List<Integer> mapVetoIds) { this.mapVetoIds = mapVetoIds; }
    public java.util.List<Integer> getGameScoreIds() { return gameScoreIds; }
    public void setGameScoreIds(java.util.List<Integer> gameScoreIds) { this.gameScoreIds = gameScoreIds; }
}
