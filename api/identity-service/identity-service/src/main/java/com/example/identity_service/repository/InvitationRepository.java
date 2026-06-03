package com.example.identity_service.repository;

import com.example.identity_service.entity.Invitation;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.UUID;

public interface InvitationRepository extends JpaRepository<Invitation, UUID> {
    List<Invitation> findAllByReceivingPlayer(UUID receivingPlayer);
    List<Invitation> findAllBySendingTeam(UUID sendingTeam);
    boolean existsBySendingTeamAndReceivingPlayer(UUID sendingTeam, UUID receivingPlayer);
    void deleteAllByReceivingPlayer(UUID receivingPlayer);
    void deleteAllBySendingAdmin(UUID sendingAdmin);
    void deleteAllBySendingTeam(UUID sendingTeam);
}
