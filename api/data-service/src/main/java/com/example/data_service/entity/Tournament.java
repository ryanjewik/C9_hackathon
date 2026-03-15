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
@Table(name = "esports_tournaments")
public class Tournament {
    @Id
    private Integer id;

    @Column(nullable = false)
    private String name;

    @Column(name = "tier")
    private String tier;

    @Column(name = "start_date")
    private String startDate;

    @Column(name = "end_date")
    private String endDate;

    @Column(name = "location")
    private String location;

    @Column(name = "prize_pool")
    private String prizePool;

    @Column(name = "status")
    private String status;

    public Tournament() {}

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getTier() { return tier; }
    public void setTier(String tier) { this.tier = tier; }

    public String getStartDate() { return startDate; }
    public void setStartDate(String startDate) { this.startDate = startDate; }

    public String getEndDate() { return endDate; }
    public void setEndDate(String endDate) { this.endDate = endDate; }

    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }

    public String getPrizePool() { return prizePool; }
    public void setPrizePool(String prizePool) { this.prizePool = prizePool; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}