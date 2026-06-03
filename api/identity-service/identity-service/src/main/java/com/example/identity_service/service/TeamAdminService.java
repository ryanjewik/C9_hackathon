package com.example.identity_service.service;

import com.example.identity_service.entity.Team;
import com.example.identity_service.entity.TeamMember;
import com.example.identity_service.entity.TeamMemberId;
import com.example.identity_service.entity.Invitation;
import com.example.identity_service.repository.TeamAdminRepository;
import com.example.identity_service.repository.TeamMemberRepository;
import com.example.identity_service.repository.InvitationRepository;
import com.example.identity_service.repository.ApiKeyRepository;
import com.example.identity_service.repository.AuthRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.dao.DataIntegrityViolationException;

import java.util.Optional;
import java.util.UUID;

@Service
public class TeamAdminService {

    private final TeamAdminRepository teamAdminRepository;
    private final TeamMemberRepository teamMemberRepository;
    private final InvitationRepository invitationRepository;
    private final ApiKeyRepository apiKeyRepository;
    private final AuthRepository authRepository;

    public TeamAdminService(TeamAdminRepository teamAdminRepository, TeamMemberRepository teamMemberRepository, InvitationRepository invitationRepository, ApiKeyRepository apiKeyRepository, AuthRepository authRepository) {
        this.teamAdminRepository = teamAdminRepository;
        this.teamMemberRepository = teamMemberRepository;
        this.invitationRepository = invitationRepository;
        this.apiKeyRepository = apiKeyRepository;
        this.authRepository = authRepository;
    }

    @Transactional
    public Team createNewTeam(String name, UUID ownerId) {
        if (name == null || name.isBlank() || ownerId == null) {
            throw new com.example.identity_service.exception.BadRequestException("invalid_team_name_or_owner");
        }

        Team team = new Team();
        team.setName(name);
        team.setOwnerUserId(ownerId);

        try {
            Team saved = teamAdminRepository.save(team);
            return saved;
        } catch (DataIntegrityViolationException ex) {
            throw new com.example.identity_service.exception.ConflictException("team_name_taken");
        }
    }

    public java.util.List<Team> viewTeams(UUID userId) {
        java.util.List<Team> out = new java.util.ArrayList<>();
        // owned teams
        out.addAll(teamAdminRepository.findAllByOwnerUserId(userId));
        // teams where member
        java.util.List<TeamMember> memberships = teamMemberRepository.findAllByIdUserId(userId);
        java.util.Set<UUID> teamIds = new java.util.HashSet<>();
        for (TeamMember m : memberships) teamIds.add(m.getId().getTeamId());
        if (!teamIds.isEmpty()) {
            out.addAll(teamAdminRepository.findAllById(teamIds));
        }
        return out;
    }

    public java.util.List<TeamMember> viewTeamMembers(UUID teamId) {
        return teamMemberRepository.findAllByIdTeamId(teamId);
    }

    /** Invite by username or email string — returns a result code + optional invitation. */
    public enum InviteResult { OK, USER_NOT_FOUND, ALREADY_MEMBER, ALREADY_INVITED, NOT_ALLOWED, TEAM_NOT_FOUND }

    public static class InviteByUsernameResult {
        public final InviteResult code;
        public final Invitation invitation;
        InviteByUsernameResult(InviteResult code, Invitation inv) { this.code = code; this.invitation = inv; }
    }

    @Transactional
    public InviteByUsernameResult inviteByUsernameOrEmail(UUID teamId, String usernameOrEmail, UUID sendingAdmin) {
        if (usernameOrEmail == null || usernameOrEmail.isBlank()) return new InviteByUsernameResult(InviteResult.USER_NOT_FOUND, null);

        // Resolve target user
        com.example.identity_service.entity.User target = null;
        if (usernameOrEmail.contains("@")) {
            target = authRepository.findByEmail(usernameOrEmail).orElse(null);
        } else {
            target = authRepository.findByUsername(usernameOrEmail).orElse(null);
        }
        if (target == null) return new InviteByUsernameResult(InviteResult.USER_NOT_FOUND, null);

        UUID targetId = target.getId();

        // Team exists?
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return new InviteByUsernameResult(InviteResult.TEAM_NOT_FOUND, null);
        Team t = tOpt.get();

        // Permission check: owner or admin
        boolean allowed = sendingAdmin.equals(t.getOwnerUserId());
        if (!allowed) {
            for (TeamMember m : teamMemberRepository.findAllByIdUserId(sendingAdmin)) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { allowed = true; break; }
            }
        }
        if (!allowed) return new InviteByUsernameResult(InviteResult.NOT_ALLOWED, null);

        // Already a member?
        if (targetId.equals(t.getOwnerUserId())) return new InviteByUsernameResult(InviteResult.ALREADY_MEMBER, null);
        for (TeamMember m : teamMemberRepository.findAllByIdUserId(targetId)) {
            if (m.getId().getTeamId().equals(teamId)) return new InviteByUsernameResult(InviteResult.ALREADY_MEMBER, null);
        }

        // Already invited?
        if (invitationRepository.existsBySendingTeamAndReceivingPlayer(teamId, targetId))
            return new InviteByUsernameResult(InviteResult.ALREADY_INVITED, null);

        Invitation inv = new Invitation(teamId, targetId, sendingAdmin);
        try {
            Invitation saved = invitationRepository.save(inv);
            return new InviteByUsernameResult(InviteResult.OK, saved);
        } catch (Exception ex) {
            return new InviteByUsernameResult(InviteResult.NOT_ALLOWED, null);
        }
    }

    @Transactional
    public Optional<Invitation> invite(UUID teamId, UUID receivingPlayer, UUID sendingAdmin) {
        // Basic checks: team exists and sendingAdmin is owner or admin member
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return Optional.empty();
        Team t = tOpt.get();
        boolean allowed = sendingAdmin.equals(t.getOwnerUserId());
        if (!allowed) {
            java.util.List<TeamMember> members = teamMemberRepository.findAllByIdUserId(sendingAdmin);
            for (TeamMember m : members) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) {
                    allowed = true; break;
                }
            }
        }
        if (!allowed) return Optional.empty();

        Invitation inv = new Invitation(teamId, receivingPlayer, sendingAdmin);
        try {
            Invitation saved = invitationRepository.save(inv);
            return Optional.of(saved);
        } catch (Exception ex) {
            return Optional.empty();
        }
    }

    @Transactional
    public boolean acceptInvite(UUID inviteId, UUID receiverId) {
        Optional<Invitation> invOpt = invitationRepository.findById(inviteId);
        if (invOpt.isEmpty()) return false;
        Invitation inv = invOpt.get();
        if (!inv.getReceivingPlayer().equals(receiverId)) return false;

        TeamMemberId id = new TeamMemberId();
        id.setTeamId(inv.getSendingTeam());
        id.setUserId(receiverId);
        TeamMember member = new TeamMember(id, "member");
        try {
            teamMemberRepository.save(member);
            invitationRepository.deleteById(inviteId);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    @Transactional
    public Optional<ApiKeyCreationResult> createApiKey(UUID teamId, String name, UUID requesterId) {
        if (teamId == null || requesterId == null || name == null || name.isBlank()) return Optional.empty();
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return Optional.empty();
        Team t = tOpt.get();
        boolean allowed = requesterId.equals(t.getOwnerUserId());
        if (!allowed) {
            java.util.List<TeamMember> members = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : members) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { allowed = true; break; }
            }
        }
        if (!allowed) return Optional.empty();

        try {
            // generate strong random key and a prefix
            byte[] rnd = new byte[32];
            java.security.SecureRandom.getInstanceStrong().nextBytes(rnd);
            String fullKey = java.util.Base64.getUrlEncoder().withoutPadding().encodeToString(rnd);
            String keyPrefix = fullKey.substring(0, Math.min(8, fullKey.length()));
            // store hash of key (one-way) - keep existing SHA-256 behavior for minimal change
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(fullKey.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) sb.append(String.format("%02x", b));
            String keyHash = sb.toString();

            com.example.identity_service.entity.ApiKey ak = new com.example.identity_service.entity.ApiKey(teamId, name, keyPrefix, keyHash);
            com.example.identity_service.entity.ApiKey saved = apiKeyRepository.save(ak);
            // Return the saved entity plus the plaintext key directly to the caller.
            // Do NOT persist or cache the plaintext anywhere.
            ApiKeyCreationResult result = new ApiKeyCreationResult(saved, fullKey);
            return Optional.of(result);
        } catch (Exception ex) {
            return Optional.empty();
        }
    }

    public java.util.List<com.example.identity_service.entity.ApiKey> listApiKeys(UUID teamId, UUID requesterId) {
        if (teamId == null || requesterId == null) return java.util.Collections.emptyList();
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return java.util.Collections.emptyList();
        Team t = tOpt.get();
        boolean allowed = requesterId.equals(t.getOwnerUserId());
        if (!allowed) {
            java.util.List<TeamMember> members = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : members) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { allowed = true; break; }
            }
        }
        if (!allowed) return java.util.Collections.emptyList();
        return apiKeyRepository.findAllByTeamId(teamId);
    }

    @Transactional
    public boolean deleteApiKey(UUID apiKeyId, UUID requesterId) {
        if (apiKeyId == null || requesterId == null) return false;
        Optional<com.example.identity_service.entity.ApiKey> akOpt = apiKeyRepository.findById(apiKeyId);
        if (akOpt.isEmpty()) return false;
        com.example.identity_service.entity.ApiKey ak = akOpt.get();
        UUID teamId = ak.getTeamId();
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return false;
        Team t = tOpt.get();
        boolean allowed = requesterId.equals(t.getOwnerUserId());
        if (!allowed) {
            java.util.List<TeamMember> members = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : members) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { allowed = true; break; }
            }
        }
        if (!allowed) return false;
        try {
            apiKeyRepository.deleteById(apiKeyId);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    public java.util.List<Invitation> viewInvites(UUID userId) {
        return invitationRepository.findAllByReceivingPlayer(userId);
    }

    @Transactional
    public boolean rejectInvite(UUID inviteId, UUID requesterId) {
        Optional<Invitation> invOpt = invitationRepository.findById(inviteId);
        if (invOpt.isEmpty()) return false;
        Invitation inv = invOpt.get();
        // allow if requester is the receiving player
        if (inv.getReceivingPlayer().equals(requesterId)) {
            try {
                invitationRepository.deleteById(inviteId);
                return true;
            } catch (Exception ex) {
                return false;
            }
        }

        // or allow if requester is owner/admin of the sending team
        UUID teamId = inv.getSendingTeam();
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return false;
        Team t = tOpt.get();
        boolean allowed = requesterId.equals(t.getOwnerUserId());
        if (!allowed) {
            java.util.List<TeamMember> members = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : members) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { allowed = true; break; }
            }
        }
        if (!allowed) return false;
        try {
            invitationRepository.deleteById(inviteId);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    @Transactional
    public boolean deleteAccount(UUID userId) {
        if (userId == null) return false;
        try {
            // remove invitations where user is sender or receiver
            invitationRepository.deleteAllByReceivingPlayer(userId);
            invitationRepository.deleteAllBySendingAdmin(userId);
            // remove memberships for this user
            teamMemberRepository.deleteAllByIdUserId(userId);
            // delete teams owned by this user and related members/invitations
            java.util.List<Team> owned = teamAdminRepository.findAllByOwnerUserId(userId);
            for (Team t : owned) {
                teamMemberRepository.deleteAllByIdTeamId(t.getId());
                invitationRepository.deleteAllBySendingTeam(t.getId());
            }
            teamAdminRepository.deleteAllByOwnerUserId(userId);
            // finally delete user
            authRepository.deleteById(userId);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    @Transactional
    public boolean removeTeamMember(UUID teamId, UUID memberUserId, UUID requesterId) {
        if (teamId == null || memberUserId == null || requesterId == null) return false;
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return false;
        Team t = tOpt.get();
        // If requester is trying to remove themselves, allow (members and admins may leave)
        if (requesterId.equals(memberUserId)) {
            // owner cannot simply leave the team
            if (requesterId.equals(t.getOwnerUserId())) return false;
            TeamMemberId selfId = new TeamMemberId(teamId, memberUserId);
            if (!teamMemberRepository.existsById(selfId)) return false;
            try {
                teamMemberRepository.deleteById(selfId);
                return true;
            } catch (Exception ex) {
                return false;
            }
        }

        // Otherwise, requester must be owner or admin. Admins may delete members but NOT admins; owners can delete anyone.
        boolean isOwner = requesterId.equals(t.getOwnerUserId());
        boolean isAdmin = false;
        if (!isOwner) {
            java.util.List<TeamMember> requesterMemberships = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : requesterMemberships) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { isAdmin = true; break; }
            }
        }
        if (!isOwner && !isAdmin) return false;

        TeamMemberId id = new TeamMemberId(teamId, memberUserId);
        if (!teamMemberRepository.existsById(id)) return false;
        // if requester is admin, ensure target is not an admin
        if (isAdmin) {
            Optional<TeamMember> targetOpt = teamMemberRepository.findById(id);
            if (targetOpt.isPresent()) {
                String targetRole = targetOpt.get().getRole();
                if ("admin".equalsIgnoreCase(targetRole) || requesterId.equals(t.getOwnerUserId())) {
                    // admins may NOT delete other admins (and cannot delete owners)
                    return false;
                }
            }
        }
        try {
            teamMemberRepository.deleteById(id);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    @Transactional
    public boolean updateMemberRole(UUID teamId, UUID memberUserId, String newRole, UUID requesterId) {
        if (teamId == null || memberUserId == null || requesterId == null || newRole == null) return false;
        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return false;
        Team t = tOpt.get();
        boolean isOwner = requesterId.equals(t.getOwnerUserId());
        boolean isAdmin = false;
        if (!isOwner) {
            java.util.List<TeamMember> requesterMemberships = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : requesterMemberships) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { isAdmin = true; break; }
            }
        }
        if (!isOwner && !isAdmin) return false;

        TeamMemberId id = new TeamMemberId(teamId, memberUserId);
        Optional<TeamMember> mOpt = teamMemberRepository.findById(id);
        if (mOpt.isEmpty()) return false;
        TeamMember member = mOpt.get();
        String currentRole = member.getRole();
        String normalizedNew = newRole == null ? null : newRole.toLowerCase();

        // Rules: admins may promote members -> admin, but cannot promote to owner, and cannot demote admins -> member.
        if (isAdmin && !isOwner) {
            if ("owner".equalsIgnoreCase(normalizedNew)) {
                return false; // admin cannot make someone owner
            }
            if ("admin".equalsIgnoreCase(currentRole) && "member".equalsIgnoreCase(normalizedNew)) {
                return false; // admin cannot demote another admin
            }
        }

        // Prevent changing the team's owner role accidentally unless requester is owner
        if (!isOwner) {
            if (memberUserId.equals(t.getOwnerUserId())) {
                return false; // only owner may change owner role
            }
        }

        member.setRole(newRole);
        try {
            teamMemberRepository.save(member);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    /**
     * Allow a member to leave the team voluntarily (members and admins). Owners may NOT leave via this call.
     */
    @Transactional
    public boolean leaveTeam(UUID teamId, UUID requesterId) {
        return removeTeamMember(teamId, requesterId, requesterId);
    }

    /**
     * Delete a team if it exists and the requester is the owner.
     * Returns true when deleted, false otherwise.
     */
    @Transactional
    public boolean deleteTeam(UUID teamId, UUID requesterId) {
        if (teamId == null || requesterId == null) return false;

        Optional<Team> tOpt = teamAdminRepository.findById(teamId);
        if (tOpt.isEmpty()) return false;
        Team t = tOpt.get();
        if (!requesterId.equals(t.getOwnerUserId())) {
            return false; // not owner
        }

        try {
            teamAdminRepository.deleteById(teamId);
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    /** Get basic info for a single team. */
    public Optional<Team> getTeamInfo(UUID teamId) {
        return teamAdminRepository.findById(teamId);
    }

    /** Return team members with their usernames resolved from the users table. */
    public java.util.List<java.util.Map<String, Object>> viewTeamMembersRich(UUID teamId) {
        java.util.List<TeamMember> members = teamMemberRepository.findAllByIdTeamId(teamId);
        java.util.List<java.util.Map<String, Object>> out = new java.util.ArrayList<>();
        for (TeamMember m : members) {
            UUID uid = m.getId().getUserId();
            String username = authRepository.findById(uid)
                .map(com.example.identity_service.entity.User::getUsername)
                .orElse(uid.toString());
            java.util.Map<String, Object> entry = new java.util.HashMap<>();
            entry.put("userId", uid.toString());
            entry.put("username", username);
            entry.put("role", m.getRole());
            entry.put("joinedAt", m.getCreatedAt() != null ? m.getCreatedAt().toString() : null);
            out.add(entry);
        }
        return out;
    }
}
