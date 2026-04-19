package com.example.identity_service.service;

import java.security.Key;
import java.security.interfaces.RSAPrivateKey;
import java.security.interfaces.RSAPublicKey;
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
    private final RSAPrivateKey rsaPrivateKey;
    private final RSAPublicKey rsaPublicKey;
    private final long expirationMs;
    private final String issuer;
    private final String audience;

    public JwtService(@Value("${jwt.secret:}") String secret,
                      @Value("${jwt.expiration-ms}") long expirationMs,
                      @Value("${jwt.issuer:}") String issuer,
                      @Value("${jwt.audience:}") String audience,
                      KeyService keyService) {
        // If RSA keys are available from KeyService, prefer RS256; otherwise fall back to HMAC
        this.rsaPrivateKey = keyService != null ? keyService.getPrivateKey() : null;
        this.rsaPublicKey = keyService != null ? keyService.getPublicKey() : null;
        if (this.rsaPrivateKey != null) {
            this.key = null;
        } else {
            if (secret == null || secret.isEmpty()) {
                throw new IllegalArgumentException("jwt.secret must be set when not using RSA keys");
            }
            this.key = Keys.hmacShaKeyFor(secret.getBytes());
        }
        this.expirationMs = expirationMs;
        this.issuer = (issuer == null || issuer.isEmpty()) ? null : issuer;
        this.audience = (audience == null || audience.isEmpty()) ? null : audience;
    }

    public String generateToken(UUID userId, String username) {
        Instant now = Instant.now();
        Date issuedAt = Date.from(now);
        Date expiry = Date.from(now.plusMillis(expirationMs));

        if (rsaPrivateKey != null) {
            io.jsonwebtoken.JwtBuilder builder = Jwts.builder()
                .setHeaderParam("kid", System.getenv().getOrDefault("JWT_KEY_ID", "dev-key"))
                .setSubject(userId.toString())
                .claim("username", username)
                .setIssuedAt(issuedAt)
                .setExpiration(expiry);
            if (this.issuer != null) builder.setIssuer(this.issuer);
            if (this.audience != null) builder.setAudience(this.audience);
            return builder.signWith(rsaPrivateKey, SignatureAlgorithm.RS256).compact();
        }

        io.jsonwebtoken.JwtBuilder hmacBuilder = Jwts.builder()
            .setSubject(userId.toString())
            .claim("username", username)
            .setIssuedAt(issuedAt)
            .setExpiration(expiry);
        if (this.issuer != null) hmacBuilder.setIssuer(this.issuer);
        if (this.audience != null) hmacBuilder.setAudience(this.audience);
        return hmacBuilder.signWith(key, SignatureAlgorithm.HS256).compact();
    }

    /**
     * Generate a token for an API key / team context. Subject will be the team id.
     * expirationMs controls TTL for this token.
     */
    public String generateTokenForApiKey(UUID teamId, String keyPrefix, long expirationMsOverride) {
        Instant now = Instant.now();
        Date issuedAt = Date.from(now);
        Date expiry = Date.from(now.plusMillis(expirationMsOverride));

        if (rsaPrivateKey != null) {
            io.jsonwebtoken.JwtBuilder builder = Jwts.builder()
                .setHeaderParam("kid", System.getenv().getOrDefault("JWT_KEY_ID", "dev-key"))
                .setSubject(teamId.toString())
                .claim("key_prefix", keyPrefix)
                .claim("apikey", true)
                .setIssuedAt(issuedAt)
                .setExpiration(expiry);
            if (this.issuer != null) builder.setIssuer(this.issuer);
            if (this.audience != null) builder.setAudience(this.audience);
            return builder.signWith(rsaPrivateKey, SignatureAlgorithm.RS256).compact();
        }

        io.jsonwebtoken.JwtBuilder hmacBuilder = Jwts.builder()
            .setSubject(teamId.toString())
            .claim("key_prefix", keyPrefix)
            .claim("apikey", true)
            .setIssuedAt(issuedAt)
            .setExpiration(expiry);
        if (this.issuer != null) hmacBuilder.setIssuer(this.issuer);
        if (this.audience != null) hmacBuilder.setAudience(this.audience);
        return hmacBuilder.signWith(key, SignatureAlgorithm.HS256).compact();
    }

    /**
     * Validate a JWT and return the user id subject if valid.
     */
    public Optional<UUID> validateTokenAndGetUserId(String token) {
        try {
            if (rsaPrivateKey != null) {
                // validate RS256 signature using public key via Nimbus decoder is handled by gateway; here we can parse without verifying if desired
                Claims claims = Jwts.parserBuilder().setSigningKey(rsaPublicKey).build().parseClaimsJws(token).getBody();
                String subj = claims.getSubject();
                if (subj == null) return Optional.empty();
                return Optional.of(UUID.fromString(subj));
            }

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
