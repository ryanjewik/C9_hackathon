package com.example.data_service.dto;

public class MapStatsDto {
    private String mapSelected;
    private Long pickCount;
    private Long banCount;

    public MapStatsDto(String mapSelected, Long pickCount, Long banCount) {
        this.mapSelected = mapSelected;
        this.pickCount = pickCount;
        this.banCount = banCount;
    }

    public String getMapSelected() { return mapSelected; }
    public void setMapSelected(String mapSelected) { this.mapSelected = mapSelected; }
    public Long getPickCount() { return pickCount; }
    public void setPickCount(Long pickCount) { this.pickCount = pickCount; }
    public Long getBanCount() { return banCount; }
    public void setBanCount(Long banCount) { this.banCount = banCount; }
}
