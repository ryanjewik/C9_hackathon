package com.example.identity_service.service;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Temporary in-memory cache to hold freshly-generated plaintext API keys so
 * the controller can return them once. Keys are removed on retrieval.
 */
public final class ApiKeyPlainCache {
    private static final Map<UUID, String> cache = new ConcurrentHashMap<>();

    public static void store(UUID id, String plaintext) { cache.put(id, plaintext); }
    public static String take(UUID id) { return cache.remove(id); }
}
