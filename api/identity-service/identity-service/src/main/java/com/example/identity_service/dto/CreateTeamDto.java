package com.example.identity_service.dto;
import java.util.UUID;

public class CreateTeamDto {
    private String name;
    private UUID id;

    public CreateTeamDto(){}

    public CreateTeamDto(String name, UUID id){
        this.name = name;
        this.id = id;
    }

    public String getName(){
        return name;
    }

    public void setName(String name){
        this.name = name;
    }

    public UUID getId(){
        return id;
    }

    public void setId(UUID id){
        this.id = id;
    }
}
