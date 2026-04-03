package com.example.identity_service.service;

import com.example.identity_service.entity.ApiKey;

/**
 * Simple result wrapper for API key creation: stored entity + one-time plaintext.
 */
public final class ApiKeyCreationResult {
    private final ApiKey apiKey;
    private final String plaintext;

    public ApiKeyCreationResult(ApiKey apiKey, String plaintext) {
        this.apiKey = apiKey;
        this.plaintext = plaintext;
    }

    public ApiKey getApiKey() { return apiKey; }
    public String getPlaintext() { return plaintext; }
}
