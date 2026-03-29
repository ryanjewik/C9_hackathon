package com.example.identity_service.service;

import com.example.identity_service.entity.Team;
import com.example.identity_service.repository.TeamAdminRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.dao.DataIntegrityViolationException;

import java.util.Optional;
import java.util.UUID;

@Service
public class TeamAdminService {

    private final TeamAdminRepository teamAdminRepository;

    public TeamAdminService(TeamAdminRepository teamAdminRepository) {
        this.teamAdminRepository = teamAdminRepository;
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
}
