package com.example.identity_service.repository;
import com.example.identity_service.entity.Team;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.UUID;
import java.util.Optional;
import java.time.OffsetDateTime;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.stereotype.Repository;

// public interface TeamAdminRepository extends JpaRepository<Team, UUID> {
//     @Modifying
//     @Transactional
//     @Query(
//         value = "INSERT INTO teams (id, name, owner_user_id, created_at) VALUES (:id, :name, :owner_user_id, :created_at)", 
//         nativeQuery = true
//     )
//     void createTeam(@Param("id") UUID id, @Param("name") String name, @Param("owner_user_id") UUID owner_user_id, @Param("created_at") OffsetDateTime now);


// }
@Repository
public interface TeamAdminRepository extends JpaRepository<Team, UUID> {
    Optional<Team> findById(UUID id);
}