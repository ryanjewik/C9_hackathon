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
@Table(name = "invitations")
public class Invitation {

    @Id
    @GeneratedValue(generator = "UUID")
    @GenericGenerator(name = "UUID", strategy = "org.hibernate.id.UUIDGenerator")
    @Column(name = "id", updatable = false, nullable = false)
    private UUID id;

    @Column(name = "sending_team", nullable = false)
    private UUID sendingTeam;

    @Column(name = "receiving_player", nullable = false)
    private UUID receivingPlayer;

    @Column(name = "sending_admin", nullable = false)
    private UUID sendingAdmin;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, columnDefinition = "TIMESTAMP WITH TIME ZONE")
    private OffsetDateTime createdAt;

    public Invitation() {}

    public Invitation(UUID sendingTeam, UUID receivingPlayer, UUID sendingAdmin) {
        this.sendingTeam = sendingTeam;
        this.receivingPlayer = receivingPlayer;
        this.sendingAdmin = sendingAdmin;
    }

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public UUID getSendingTeam() {
        return sendingTeam;
    }

    public void setSendingTeam(UUID sendingTeam) {
        this.sendingTeam = sendingTeam;
    }

    public UUID getReceivingPlayer() {
        return receivingPlayer;
    }

    public void setReceivingPlayer(UUID receivingPlayer) {
        this.receivingPlayer = receivingPlayer;
    }

    public UUID getSendingAdmin() {
        return sendingAdmin;
    }

    public void setSendingAdmin(UUID sendingAdmin) {
        this.sendingAdmin = sendingAdmin;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(OffsetDateTime createdAt) {
        this.createdAt = createdAt;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Invitation that = (Invitation) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Invitation{" +
                "id=" + id +
                ", sendingTeam=" + sendingTeam +
                ", receivingPlayer=" + receivingPlayer +
                ", sendingAdmin=" + sendingAdmin +
                ", createdAt=" + createdAt +
                '}';
    }
}
