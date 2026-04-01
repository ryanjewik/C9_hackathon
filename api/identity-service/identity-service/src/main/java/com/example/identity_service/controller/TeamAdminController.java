package com.example.identity_service.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import com.example.identity_service.service.TeamAdminService;
import com.example.identity_service.dto.CreateTeamDto;
import com.example.identity_service.dto.InviteDto;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.jwt.Jwt;
import java.util.UUID;
import java.util.Map;
import java.util.Optional;
import com.example.identity_service.entity.Team;
import com.example.identity_service.entity.Invitation;
import com.example.identity_service.entity.TeamMember;
import com.example.identity_service.dto.TeamMemberRoleDto;
import com.example.identity_service.entity.Invitation;
import com.example.identity_service.entity.TeamMember;


@RestController
@RequestMapping("teamadmin")
public class TeamAdminController {
    private final TeamAdminService teamAdminService;

    public TeamAdminController(TeamAdminService teamAdminService){
        this.teamAdminService = teamAdminService;
    }

    /**
     * Create a team. Owner is inferred from the authenticated JWT subject.
     * Requires a valid Bearer JWT (configured via the resource-server).
     */
    @PostMapping("/create")
    public ResponseEntity<?> createNewTeam(@AuthenticationPrincipal Jwt jwt, @RequestBody CreateTeamDto team){
        if (jwt == null) {
            return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        }
        String sub = jwt.getSubject();
        if (sub == null) {
            return ResponseEntity.status(401).body(Map.of("error", "invalid_token"));
        }

        UUID ownerId;
        try {
            ownerId = UUID.fromString(sub);
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(401).body(Map.of("error", "invalid_token_subject"));
        }

        String name = team.getName();
        Optional<Team> createdTeam = teamAdminService.createNewTeam(name, ownerId);
        if (createdTeam.isEmpty()){
            return ResponseEntity.status(400).body(Map.of("error", "failed_to_create_team"));
        }
        Team t = createdTeam.get();
        return ResponseEntity.status(201).body(Map.of(
            "team_id", t.getId(),
            "team_name", t.getName(),
            "creation_time", t.getCreatedAt()
        ));
    }

    /**
     * Delete a team. Only the owner (token subject) can delete.
     */
    @DeleteMapping("/delete/{teamId}")
    public ResponseEntity<?> deleteTeam(@PathVariable("teamId") UUID teamId,
                                        @AuthenticationPrincipal Jwt jwt){
        if (jwt == null) {
            return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        }
        UUID requester;
        try {
            requester = UUID.fromString(jwt.getSubject());
        } catch (Exception ex) {
            return ResponseEntity.status(401).body(Map.of("error", "invalid_token_subject"));
        }

        boolean ok = teamAdminService.deleteTeam(teamId, requester);
        if (!ok) {
            return ResponseEntity.status(403).body(Map.of("error", "not_authorized_or_not_found"));
        }
        return ResponseEntity.noContent().build();
    }

    /** Delete the authenticated user's account and related data. */
    @DeleteMapping("/account")
    public ResponseEntity<?> deleteAccount(@AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        boolean ok = teamAdminService.deleteAccount(userId);
        if (!ok) return ResponseEntity.status(500).body(Map.of("error","failed_to_delete_account"));
        return ResponseEntity.noContent().build();
    }

    /** View teams the authenticated user owns or is a member of */
    @org.springframework.web.bind.annotation.GetMapping("/teams")
    public ResponseEntity<?> viewTeams(@AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        return ResponseEntity.ok(teamAdminService.viewTeams(userId));
    }

    /** View members of a given team */
    @org.springframework.web.bind.annotation.GetMapping("/{teamId}/members")
    public ResponseEntity<?> viewTeamMembers(@PathVariable("teamId") UUID teamId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        return ResponseEntity.ok(teamAdminService.viewTeamMembers(teamId));
    }

    /** Send an invitation to a player to join a team */
    @PostMapping("/invite")
    public ResponseEntity<?> invite(@AuthenticationPrincipal Jwt jwt, @RequestBody InviteDto dto) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID sender;
        try { sender = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        Optional<Invitation> inv = teamAdminService.invite(dto.getSendingTeam(), dto.getReceivingPlayer(), sender);
        if (inv.isEmpty()) return ResponseEntity.status(400).body(Map.of("error","failed_to_create_invite"));
        return ResponseEntity.status(201).body(inv.get());
    }

    /** Accept an invitation */
    @PostMapping("/invite/{inviteId}/accept")
    public ResponseEntity<?> acceptInvite(@PathVariable("inviteId") UUID inviteId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID receiver;
        try { receiver = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        boolean ok = teamAdminService.acceptInvite(inviteId, receiver);
        if (!ok) return ResponseEntity.status(400).body(Map.of("error","failed_to_accept_invite"));
        return ResponseEntity.noContent().build();
    }

    /** View invitations sent to the authenticated user */
    @org.springframework.web.bind.annotation.GetMapping("/invites")
    public ResponseEntity<?> viewInvites(@AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        return ResponseEntity.ok(teamAdminService.viewInvites(userId));
    }

    /** Reject (delete) an invitation */
    @DeleteMapping("/invite/{inviteId}")
    public ResponseEntity<?> rejectInvite(@PathVariable("inviteId") UUID inviteId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        boolean ok = teamAdminService.rejectInvite(inviteId, userId);
        if (!ok) return ResponseEntity.status(400).body(Map.of("error","failed_to_reject_invite"));
        return ResponseEntity.noContent().build();
    }

    /** Remove a member from a team (owner or admin can remove). */
    @DeleteMapping("/{teamId}/members/{memberId}")
    public ResponseEntity<?> removeTeamMember(@PathVariable("teamId") UUID teamId,
                                              @PathVariable("memberId") UUID memberId,
                                              @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID requester;
        try { requester = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        boolean ok = teamAdminService.removeTeamMember(teamId, memberId, requester);
        if (!ok) return ResponseEntity.status(403).body(Map.of("error","failed_to_remove_member"));
        return ResponseEntity.noContent().build();
    }

    /** Update a member's role */
    @PatchMapping("/{teamId}/members/{memberId}/role")
    public ResponseEntity<?> updateMemberRole(@PathVariable("teamId") UUID teamId,
                                              @PathVariable("memberId") UUID memberId,
                                              @RequestBody TeamMemberRoleDto dto,
                                              @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID requester;
        try { requester = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        boolean ok = teamAdminService.updateMemberRole(teamId, memberId, dto.getRole(), requester);
        if (!ok) return ResponseEntity.status(403).body(Map.of("error","failed_to_update_role"));
        return ResponseEntity.noContent().build();
    }

    /** Get members for a team (duplicate of viewTeamMembers but explicit path) */
    @GetMapping("/{teamId}/members/list")
    public ResponseEntity<?> getTeamMembers(@PathVariable("teamId") UUID teamId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        return ResponseEntity.ok(teamAdminService.viewTeamMembers(teamId));
    }

}
