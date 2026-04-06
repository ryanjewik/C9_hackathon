package com.example.identity_service.dto;

public class ApiKeyTokenRequestDto {
    private String key;
    private String audience;

    public ApiKeyTokenRequestDto() {}

    public String getKey() { return key; }
    public void setKey(String key) { this.key = key; }

    public String getAudience() { return audience; }
    public void setAudience(String audience) { this.audience = audience; }
}
