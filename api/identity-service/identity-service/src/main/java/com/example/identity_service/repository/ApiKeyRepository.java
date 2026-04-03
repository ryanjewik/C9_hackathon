package com.example.identity_service.repository;

import com.example.identity_service.entity.ApiKey;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.UUID;

public interface ApiKeyRepository extends JpaRepository<ApiKey, UUID> {
    List<ApiKey> findAllByTeamId(UUID teamId);
    void deleteAllByTeamId(UUID teamId);
}
