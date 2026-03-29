package com.example.identity_service.entity;

import java.time.OffsetDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

import org.hibernate.annotations.CreationTimestamp;

@Entity
@Table(name = "team_members")
public class TeamMember {

    @EmbeddedId
    private TeamMemberId id;

    @Column(name = "role", nullable = false)
    private String role = "member";

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, columnDefinition = "TIMESTAMP WITH TIME ZONE")
    private OffsetDateTime createdAt;

    public TeamMember() {}

    public TeamMember(TeamMemberId id, String role) {
        this.id = id;
        this.role = role;
    }

    public TeamMemberId getId() {
        return id;
    }

    public void setId(TeamMemberId id) {
        this.id = id;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(OffsetDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
