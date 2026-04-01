package com.example.identity_service.repository;

import com.example.identity_service.entity.TeamMember;
import com.example.identity_service.entity.TeamMemberId;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.UUID;

public interface TeamMemberRepository extends JpaRepository<TeamMember, TeamMemberId> {
    List<TeamMember> findAllByIdUserId(UUID userId);
    List<TeamMember> findAllByIdTeamId(UUID teamId);
    void deleteAllByIdUserId(UUID userId);
    void deleteAllByIdTeamId(UUID teamId);
}
