package com.example.identity_service.entity;

import java.time.OffsetDateTime;
import java.util.Objects;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.GenericGenerator;

@Entity
@Table(name = "vods")
public class Vod {

    @Id
    @GeneratedValue(generator = "UUID")
    @GenericGenerator(name = "UUID", strategy = "org.hibernate.id.UUIDGenerator")
    @Column(name = "id", updatable = false, nullable = false)
    private UUID id;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "team_id", nullable = false)
    private UUID teamId;

    @Column(name = "video_link", nullable = false)
    private String videoLink;

    @Column(name = "status", nullable = false)
    private String status = "uploaded";

    @Column(name = "result_path")
    private String resultPath;

    @CreationTimestamp
    @Column(name = "uploaded_at", nullable = false, columnDefinition = "TIMESTAMP WITH TIME ZONE")
    private OffsetDateTime uploadedAt;

    public Vod() {}

    public Vod(String name, UUID teamId, String videoLink) {
        this.name = name;
        this.teamId = teamId;
        this.videoLink = videoLink;
    }

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public UUID getTeamId() {
        return teamId;
    }

    public void setTeamId(UUID teamId) {
        this.teamId = teamId;
    }

    public String getVideoLink() {
        return videoLink;
    }

    public void setVideoLink(String videoLink) {
        this.videoLink = videoLink;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getResultPath() {
        return resultPath;
    }

    public void setResultPath(String resultPath) {
        this.resultPath = resultPath;
    }

    public OffsetDateTime getUploadedAt() {
        return uploadedAt;
    }

    public void setUploadedAt(OffsetDateTime uploadedAt) {
        this.uploadedAt = uploadedAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Vod vod = (Vod) o;
        return Objects.equals(id, vod.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
