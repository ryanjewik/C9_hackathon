package com.example.identity_service.dto;

import java.util.UUID;

public class InviteDto {
    private UUID sendingTeam;
    private UUID receivingPlayer;

    public UUID getSendingTeam() { return sendingTeam; }
    public void setSendingTeam(UUID sendingTeam) { this.sendingTeam = sendingTeam; }
    public UUID getReceivingPlayer() { return receivingPlayer; }
    public void setReceivingPlayer(UUID receivingPlayer) { this.receivingPlayer = receivingPlayer; }
}
