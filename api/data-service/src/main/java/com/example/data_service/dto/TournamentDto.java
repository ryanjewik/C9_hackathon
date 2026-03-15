package com.example.data_service.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class TournamentDto {
	private Integer id;
	private String name;
	private String tier;
	@JsonProperty("start_date")
	private String startDate;
	@JsonProperty("end_date")
	private String endDate;
	private String location;
	@JsonProperty("prize_pool")
	private String prizePool;
	private String status;

	public TournamentDto() {}

	public TournamentDto(Integer id, String name, String tier, String startDate, String endDate, String location, String prizePool, String status) {
		this.id = id;
		this.name = name;
		this.tier = tier;
		this.startDate = startDate;
		this.endDate = endDate;
		this.location = location;
		this.prizePool = prizePool;
		this.status = status;
	}

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
