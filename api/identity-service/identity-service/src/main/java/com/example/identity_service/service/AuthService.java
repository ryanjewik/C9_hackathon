package com.example.identity_service.service;
import org.springframework.stereotype.Service;
import com.example.identity_service.entity.User;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.Date;
import java.util.Optional;
import com.example.identity_service.repository.AuthRepository;
import com.example.identity_service.repository.ApiKeyRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import java.util.UUID;

@Service
public class AuthService {
    private final AuthRepository loginRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private final ApiKeyRepository apiKeyRepository;
    private final long apiKeyTokenTtlMs;

    public AuthService(AuthRepository loginRepository, PasswordEncoder passwordEncoder, JwtService jwtService, ApiKeyRepository apiKeyRepository,
                       @org.springframework.beans.factory.annotation.Value("${jwt.apikey-expiration-ms:900000}") long apiKeyTokenTtlMs){
        this.loginRepository = loginRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
        this.apiKeyRepository = apiKeyRepository;
        this.apiKeyTokenTtlMs = apiKeyTokenTtlMs;
    }

    /**
     * Authenticate the user by username/password and return a JWT if successful.
     */
    public String authenticate(String username, String password){
        Optional<User> userOpt = loginRepository.findByUsername(username);
        if (userOpt.isEmpty()){
            throw new com.example.identity_service.exception.UnauthorizedException("invalid_credentials");
        }
        User user = userOpt.get();
        if (!passwordEncoder.matches(password, user.getPasswordHash())){
            throw new com.example.identity_service.exception.UnauthorizedException("invalid_credentials");
        }

        String token = jwtService.generateToken(user.getId(), user.getUsername());
        return token;
    }

    /**
     * Register a new user. Returns Optional JWT when successful.
     * If username or email already exists, returns Optional.empty().
     */
    public String register(String username, String email, String password){
        if (username == null || username.isBlank() || email == null || email.isBlank() || password == null || password.isBlank()){
            throw new com.example.identity_service.exception.BadRequestException("username_email_password_required");
        }

        if (loginRepository.findByUsername(username).isPresent()){
            throw new com.example.identity_service.exception.ConflictException("username_taken");
        }
        if (loginRepository.findByEmail(email).isPresent()){
            throw new com.example.identity_service.exception.ConflictException("email_taken");
        }

        String hash = passwordEncoder.encode(password);
        OffsetDateTime now = OffsetDateTime.now();
        User user = new User(username, email, hash, now);
        User saved = loginRepository.save(user);

        String token = jwtService.generateToken(saved.getId(), saved.getUsername());
        return token;
    }

    /**
     * Validate a plaintext API key and issue a short-lived JWT scoped for data access.
     * Returns a JWT string when successful, or throws Unauthorized/NotFound.
     */
    public String tokenForApiKey(String plaintextKey){
        if (plaintextKey == null || plaintextKey.isBlank()) throw new com.example.identity_service.exception.BadRequestException("api_key_required");

        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(plaintextKey.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) sb.append(String.format("%02x", b));
            String keyHash = sb.toString();

            java.util.Optional<com.example.identity_service.entity.ApiKey> akOpt = apiKeyRepository.findByKeyHash(keyHash);
            if (akOpt.isEmpty()) throw new com.example.identity_service.exception.UnauthorizedException("invalid_api_key");
            com.example.identity_service.entity.ApiKey ak = akOpt.get();
            if (!"active".equalsIgnoreCase(ak.getStatus())) throw new com.example.identity_service.exception.ForbiddenException("api_key_inactive");

            // update last used timestamp
            ak.setLastUsedAt(java.time.OffsetDateTime.now());
            apiKeyRepository.save(ak);

            // Issue a JWT whose subject is the team id and includes key_prefix
            java.util.UUID teamId = ak.getTeamId();
            String token = jwtService.generateTokenForApiKey(teamId, ak.getKeyPrefix(), apiKeyTokenTtlMs);
            return token;
        } catch (Exception ex) {
            throw new com.example.identity_service.exception.RegistrationFailedException();
        }
    }
}
