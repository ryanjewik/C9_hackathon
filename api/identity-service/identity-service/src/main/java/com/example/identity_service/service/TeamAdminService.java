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
    public Optional<Team> createNewTeam(String name, UUID ownerId) {
        if (name == null || name.isBlank() || ownerId == null) {
            return Optional.empty();
        }

        Team team = new Team();
        team.setName(name);
        team.setOwnerUserId(ownerId);

        try {
            Team saved = teamAdminRepository.save(team);
            return Optional.of(saved);
        } catch (DataIntegrityViolationException ex) {
            return Optional.empty();
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
    public Optional<com.example.identity_service.entity.ApiKey> createApiKey(UUID teamId, String name, UUID requesterId) {
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
            // store hash of key
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(fullKey.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) sb.append(String.format("%02x", b));
            String keyHash = sb.toString();

            com.example.identity_service.entity.ApiKey ak = new com.example.identity_service.entity.ApiKey(teamId, name, keyPrefix, keyHash);
            com.example.identity_service.entity.ApiKey saved = apiKeyRepository.save(ak);
            // return saved entity along with the plaintext key in the caller's flow
            // We cannot attach the plaintext to the entity stored; caller should receive it separately.
            // To pass it back, we'll set a transient field is not present; instead return the saved entity and caller will reconstruct response using fullKey.
            // We'll return the entity and the caller (controller) will produce ApiKeyResponseDto with fullKey.
            // For now, store fullKey in a ThreadLocal map so controller can retrieve it immediately after creation.
            ApiKeyPlainCache.store(saved.getId(), fullKey);
            return Optional.of(saved);
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
        boolean allowed = requesterId.equals(t.getOwnerUserId());
        if (!allowed) {
            java.util.List<TeamMember> members = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : members) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { allowed = true; break; }
            }
        }
        if (!allowed) return false;

        TeamMemberId id = new TeamMemberId(teamId, memberUserId);
        if (!teamMemberRepository.existsById(id)) return false;
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
        boolean allowed = requesterId.equals(t.getOwnerUserId());
        if (!allowed) {
            java.util.List<TeamMember> members = teamMemberRepository.findAllByIdUserId(requesterId);
            for (TeamMember m : members) {
                if (m.getId().getTeamId().equals(teamId) && "admin".equalsIgnoreCase(m.getRole())) { allowed = true; break; }
            }
        }
        if (!allowed) return false;

        TeamMemberId id = new TeamMemberId(teamId, memberUserId);
        Optional<TeamMember> mOpt = teamMemberRepository.findById(id);
        if (mOpt.isEmpty()) return false;
        TeamMember member = mOpt.get();
        member.setRole(newRole);
        try {
            teamMemberRepository.save(member);
            return true;
        } catch (Exception ex) {
            return false;
        }
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
}
