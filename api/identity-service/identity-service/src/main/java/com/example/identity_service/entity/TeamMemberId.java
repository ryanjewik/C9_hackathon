package com.example.identity_service.entity;

import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;

@Embeddable
public class TeamMemberId implements Serializable {

    private static final long serialVersionUID = 1L;

    @Column(name = "team_id")
    private UUID teamId;

    @Column(name = "user_id")
    private UUID userId;

    public TeamMemberId() {}

    public TeamMemberId(UUID teamId, UUID userId) {
        this.teamId = teamId;
        this.userId = userId;
    }

    public UUID getTeamId() {
        return teamId;
    }

    public void setTeamId(UUID teamId) {
        this.teamId = teamId;
    }

    public UUID getUserId() {
        return userId;
    }

    public void setUserId(UUID userId) {
        this.userId = userId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TeamMemberId that = (TeamMemberId) o;
        return Objects.equals(teamId, that.teamId) && Objects.equals(userId, that.userId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(teamId, userId);
    }
}
