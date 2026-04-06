package com.example.identity_service.service;

import java.security.Key;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.Date;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.SignatureException;
import java.util.Optional;

import io.jsonwebtoken.JwtParser;

@Service
public class JwtService {

    private final Key key;
    private final long expirationMs;

    public JwtService(@Value("${jwt.secret}") String secret,
                      @Value("${jwt.expiration-ms}") long expirationMs) {
        // secret should be a sufficiently long base64/string; use HMAC key from bytes
        this.key = Keys.hmacShaKeyFor(secret.getBytes());
        this.expirationMs = expirationMs;
    }

    public String generateToken(UUID userId, String username) {
        Instant now = Instant.now();
        Date issuedAt = Date.from(now);
        Date expiry = Date.from(now.plusMillis(expirationMs));

        return Jwts.builder()
                .setSubject(userId.toString())
                .claim("username", username)
                .setIssuedAt(issuedAt)
                .setExpiration(expiry)
                .signWith(key, SignatureAlgorithm.HS256)
                .compact();
    }

    /**
     * Generate a token for an API key / team context. Subject will be the team id.
     * expirationMs controls TTL for this token.
     */
    public String generateTokenForApiKey(UUID teamId, String keyPrefix, long expirationMsOverride) {
        Instant now = Instant.now();
        Date issuedAt = Date.from(now);
        Date expiry = Date.from(now.plusMillis(expirationMsOverride));

        return Jwts.builder()
                .setSubject(teamId.toString())
                .claim("key_prefix", keyPrefix)
                .claim("apikey", true)
                .setIssuedAt(issuedAt)
                .setExpiration(expiry)
                .signWith(key, SignatureAlgorithm.HS256)
                .compact();
    }

    /**
     * Validate a JWT and return the user id subject if valid.
     */
    public Optional<UUID> validateTokenAndGetUserId(String token) {
        try {
            JwtParser parser = Jwts.parserBuilder().setSigningKey(key).build();
            Claims claims = parser.parseClaimsJws(token).getBody();
            String subj = claims.getSubject();
            if (subj == null) return Optional.empty();
            return Optional.of(UUID.fromString(subj));
        } catch (JwtException | IllegalArgumentException ex) {
            return Optional.empty();
        }
    }
}
