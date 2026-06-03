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
import com.example.identity_service.dto.ApiKeyCreateDto;
import com.example.identity_service.dto.ApiKeyResponseDto;
import com.example.identity_service.entity.Invitation;
import com.example.identity_service.entity.TeamMember;


@RestController
@RequestMapping("teamadmin")
public class TeamAdminController {
    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(TeamAdminController.class);
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
        log.info("Create team: ownerId={} name={}", ownerId, name);
        Team t = teamAdminService.createNewTeam(name, ownerId);
        log.info("Team created: teamId={} name={}", t.getId(), t.getName());
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

        log.info("Delete team: teamId={} requester={}", teamId, requester);
        boolean ok = teamAdminService.deleteTeam(teamId, requester);
        if (!ok) {
            log.warn("Delete team failed: teamId={} requester={}", teamId, requester);
            return ResponseEntity.status(403).body(Map.of("error", "not_authorized_or_not_found"));
        }
        log.info("Team deleted: teamId={}", teamId);
        return ResponseEntity.noContent().build();
    }

    /** Delete the authenticated user's account and related data. */
    @DeleteMapping("/account")
    public ResponseEntity<?> deleteAccount(@AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("Delete account: userId={}", userId);
        boolean ok = teamAdminService.deleteAccount(userId);
        if (!ok) return ResponseEntity.status(500).body(Map.of("error","failed_to_delete_account"));
        log.info("Account deleted: userId={}", userId);
        return ResponseEntity.noContent().build();
    }

    /** View teams the authenticated user owns or is a member of */
    @org.springframework.web.bind.annotation.GetMapping("/teams")
    public ResponseEntity<?> viewTeams(@AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("View teams: userId={}", userId);
        return ResponseEntity.ok(teamAdminService.viewTeams(userId));
    }

    /** View members of a given team */
    @org.springframework.web.bind.annotation.GetMapping("/{teamId}/members")
    public ResponseEntity<?> viewTeamMembers(@PathVariable("teamId") UUID teamId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        log.info("View team members: teamId={}", teamId);
        return ResponseEntity.ok(teamAdminService.viewTeamMembers(teamId));
    }

    /** Send an invitation by username or email to a specific team */
    @PostMapping("/{teamId}/invite")
    public ResponseEntity<?> inviteByUsernameOrEmail(
            @PathVariable("teamId") UUID teamId,
            @AuthenticationPrincipal Jwt jwt,
            @RequestBody Map<String, String> body) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID sender;
        try { sender = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error", "invalid_token_subject")); }
        String usernameOrEmail = body.get("usernameOrEmail");
        if (usernameOrEmail == null || usernameOrEmail.isBlank())
            return ResponseEntity.status(400).body(Map.of("error", "user_not_found"));

        TeamAdminService.InviteByUsernameResult result = teamAdminService.inviteByUsernameOrEmail(teamId, usernameOrEmail, sender);
        return switch (result.code) {
            case OK -> ResponseEntity.status(201).body(result.invitation);
            case USER_NOT_FOUND -> ResponseEntity.status(404).body(Map.of("error", "user_not_found"));
            case ALREADY_MEMBER -> ResponseEntity.status(409).body(Map.of("error", "already_member"));
            case ALREADY_INVITED -> ResponseEntity.status(409).body(Map.of("error", "already_invited"));
            case TEAM_NOT_FOUND -> ResponseEntity.status(404).body(Map.of("error", "team_not_found"));
            case NOT_ALLOWED -> ResponseEntity.status(403).body(Map.of("error", "forbidden"));
        };
    }

    /** Send an invitation to a player to join a team */
    @PostMapping("/invite")
    public ResponseEntity<?> invite(@AuthenticationPrincipal Jwt jwt, @RequestBody InviteDto dto) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID sender;
        try { sender = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("Send invite: team={} to={} from={}", dto.getSendingTeam(), dto.getReceivingPlayer(), sender);
        Optional<Invitation> inv = teamAdminService.invite(dto.getSendingTeam(), dto.getReceivingPlayer(), sender);
        if (inv.isEmpty()) {
            log.warn("Send invite failed: team={} to={}", dto.getSendingTeam(), dto.getReceivingPlayer());
            return ResponseEntity.status(400).body(Map.of("error","failed_to_create_invite"));
        }
        log.info("Invite created: inviteId={}", inv.get().getId());
        return ResponseEntity.status(201).body(inv.get());
    }

    /** Accept an invitation */
    @PostMapping("/invite/{inviteId}/accept")
    public ResponseEntity<?> acceptInvite(@PathVariable("inviteId") UUID inviteId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID receiver;
        try { receiver = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("Accept invite: inviteId={} receiver={}", inviteId, receiver);
        boolean ok = teamAdminService.acceptInvite(inviteId, receiver);
        if (!ok) {
            log.warn("Accept invite failed: inviteId={} receiver={}", inviteId, receiver);
            return ResponseEntity.status(400).body(Map.of("error","failed_to_accept_invite"));
        }
        log.info("Invite accepted: inviteId={}", inviteId);
        return ResponseEntity.noContent().build();
    }

    /** View invitations sent to the authenticated user */
    @org.springframework.web.bind.annotation.GetMapping("/invites")
    public ResponseEntity<?> viewInvites(@AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("View invites: userId={}", userId);
        return ResponseEntity.ok(teamAdminService.viewInvites(userId));
    }

    /** Reject (delete) an invitation */
    @DeleteMapping("/invite/{inviteId}")
    public ResponseEntity<?> rejectInvite(@PathVariable("inviteId") UUID inviteId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID userId;
        try { userId = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("Reject invite: inviteId={} userId={}", inviteId, userId);
        boolean ok = teamAdminService.rejectInvite(inviteId, userId);
        if (!ok) return ResponseEntity.status(400).body(Map.of("error","failed_to_reject_invite"));
        log.info("Invite rejected: inviteId={}", inviteId);
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
        log.info("Remove team member: teamId={} memberId={} requester={}", teamId, memberId, requester);
        boolean ok = teamAdminService.removeTeamMember(teamId, memberId, requester);
        if (!ok) {
            log.warn("Remove team member failed: teamId={} memberId={}", teamId, memberId);
            return ResponseEntity.status(403).body(Map.of("error","failed_to_remove_member"));
        }
        return ResponseEntity.noContent().build();
    }

    /** Current user leaves a team (members and admins may leave; owners may not). */
    @DeleteMapping("/{teamId}/members/me")
    public ResponseEntity<?> leaveTeam(@PathVariable("teamId") UUID teamId,
                                       @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID requester;
        try { requester = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("Leave team: teamId={} requester={}", teamId, requester);
        boolean ok = teamAdminService.leaveTeam(teamId, requester);
        if (!ok) return ResponseEntity.status(400).body(Map.of("error","failed_to_leave_team"));
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
        log.info("Update member role: teamId={} memberId={} role={} requester={}", teamId, memberId, dto.getRole(), requester);
        boolean ok = teamAdminService.updateMemberRole(teamId, memberId, dto.getRole(), requester);
        if (!ok) return ResponseEntity.status(403).body(Map.of("error","failed_to_update_role"));
        return ResponseEntity.noContent().build();
    }

    /** Get members for a team (duplicate of viewTeamMembers but explicit path) */
    @GetMapping("/{teamId}/members/list")
    public ResponseEntity<?> getTeamMembers(@PathVariable("teamId") UUID teamId, @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        log.info("Get team members list: teamId={}", teamId);
        return ResponseEntity.ok(teamAdminService.viewTeamMembers(teamId));
    }

    /** Create an API key for a team (owner or admin only). Returns the plaintext key once. */
    @PostMapping("/{teamId}/apikeys")
    public ResponseEntity<?> createApiKey(@PathVariable("teamId") UUID teamId,
                                         @RequestBody ApiKeyCreateDto dto,
                                         @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID requester;
        try { requester = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("Create API key: teamId={} name={} requester={}", teamId, dto.getName(), requester);
        var resultOpt = teamAdminService.createApiKey(teamId, dto.getName(), requester);
        if (resultOpt.isEmpty()) {
            log.warn("Create API key failed: teamId={} requester={}", teamId, requester);
            return ResponseEntity.status(403).body(Map.of("error","not_authorized_or_failed"));
        }
        var result = resultOpt.get();
        com.example.identity_service.entity.ApiKey saved = result.getApiKey();
        String plaintext = result.getPlaintext();
        ApiKeyResponseDto resp = new ApiKeyResponseDto();
        resp.setId(saved.getId());
        resp.setName(saved.getName());
        resp.setKeyPrefix(saved.getKeyPrefix());
        resp.setCreatedAt(saved.getCreatedAt());
        // return plaintext exactly once in this response; do NOT store or log it
        resp.setKey(plaintext);
        log.info("API key created: teamId={} keyId={} prefix={}", teamId, saved.getId(), saved.getKeyPrefix());
        return ResponseEntity.status(201).body(resp);
    }

    /** List API keys for a team (owner or admin). Does not reveal plaintext keys. */
    @GetMapping("/{teamId}/apikeys")
    public ResponseEntity<?> listApiKeys(@PathVariable("teamId") UUID teamId,
                                        @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID requester;
        try { requester = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("List API keys: teamId={} requester={}", teamId, requester);
        java.util.List<com.example.identity_service.entity.ApiKey> keys = teamAdminService.listApiKeys(teamId, requester);
        java.util.List<ApiKeyResponseDto> out = new java.util.ArrayList<>();
        for (com.example.identity_service.entity.ApiKey k : keys) {
            ApiKeyResponseDto dto = new ApiKeyResponseDto();
            dto.setId(k.getId());
            dto.setName(k.getName());
            dto.setKeyPrefix(k.getKeyPrefix());
            dto.setCreatedAt(k.getCreatedAt());
            // do NOT set plaintext `key` field
            out.add(dto);
        }
        return ResponseEntity.ok(out);
    }

    /** Delete an API key (owner or admin). */
    @DeleteMapping("/{teamId}/apikeys/{apiKeyId}")
    public ResponseEntity<?> deleteApiKey(@PathVariable("teamId") UUID teamId,
                                         @PathVariable("apiKeyId") UUID apiKeyId,
                                         @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        UUID requester;
        try { requester = UUID.fromString(jwt.getSubject()); } catch (Exception ex) { return ResponseEntity.status(401).body(Map.of("error","invalid_token_subject")); }
        log.info("Delete API key: teamId={} keyId={} requester={}", teamId, apiKeyId, requester);
        boolean ok = teamAdminService.deleteApiKey(apiKeyId, requester);
        if (!ok) {
            log.warn("Delete API key failed: teamId={} keyId={} requester={}", teamId, apiKeyId, requester);
            return ResponseEntity.status(403).body(Map.of("error","not_authorized_or_not_found"));
        }
        return ResponseEntity.noContent().build();
    }

    /** Get basic info for a single team (name, owner, created_at). */
    @GetMapping("/{teamId}")
    public ResponseEntity<?> getTeamInfo(@PathVariable("teamId") UUID teamId,
                                         @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        return teamAdminService.getTeamInfo(teamId)
            .map(t -> ResponseEntity.ok((Object) Map.of(
                "id", t.getId().toString(),
                "name", t.getName(),
                "ownerUserId", t.getOwnerUserId().toString(),
                "createdAt", t.getCreatedAt().toString())))
            .orElse(ResponseEntity.status(404).body(Map.of("error", "team_not_found")));
    }

    /** Get team members with resolved usernames. */
    @GetMapping("/{teamId}/members/rich")
    public ResponseEntity<?> getTeamMembersRich(@PathVariable("teamId") UUID teamId,
                                                @AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) return ResponseEntity.status(401).body(Map.of("error", "authentication_required"));
        log.info("Get rich team members: teamId={}", teamId);
        return ResponseEntity.ok(teamAdminService.viewTeamMembersRich(teamId));
    }

}
