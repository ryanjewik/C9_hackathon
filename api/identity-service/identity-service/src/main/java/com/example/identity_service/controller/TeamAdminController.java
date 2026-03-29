package com.example.identity_service.controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import com.example.identity_service.service.TeamAdminService;
import com.example.identity_service.dto.CreateTeamDto;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import java.util.UUID;
import java.util.Map;
import java.util.Optional;
import com.example.identity_service.entity.Team;


@RestController
@RequestMapping("teamadmin")
public class TeamAdminController {
    private final TeamAdminService teamAdminService;

    public TeamAdminController(TeamAdminService teamAdminService){
        this.teamAdminService = teamAdminService;
    }

    @PostMapping("/create")
    public ResponseEntity<?> createNewTeam(@RequestBody CreateTeamDto team){
        String name = team.getName();
        UUID id = team.getId();
        Optional<Team> createdTeam = teamAdminService.createNewTeam(name, id);
        if (createdTeam.isEmpty()){
            return ResponseEntity.status(400).body(Map.of("error", "failed_to_create_team"));
        }
        Team t = createdTeam.get();
        return ResponseEntity.ok(Map.of(
            "team_id", t.getId(),
            "team_name", t.getName(),
            "creation_time", t.getCreatedAt()
        ));
    }

}
