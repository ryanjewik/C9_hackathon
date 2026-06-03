package com.example.identity_service.service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
    private static final Logger log = LoggerFactory.getLogger(AuthService.class);
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
            log.warn("Login failed: user not found username={}", username);
            throw new com.example.identity_service.exception.UnauthorizedException("invalid_credentials");
        }
        User user = userOpt.get();
        if (!passwordEncoder.matches(password, user.getPasswordHash())){
            log.warn("Login failed: invalid password username={}", username);
            throw new com.example.identity_service.exception.UnauthorizedException("invalid_credentials");
        }

        log.info("Login success: userId={} username={}", user.getId(), user.getUsername());
        String token = jwtService.generateToken(user.getId(), user.getUsername());
        return token;
    }

    /**
     * Register a new user. Returns Optional JWT when successful.
     * If username or email already exists, returns Optional.empty().
     */
    public String register(String username, String email, String password){
        if (username == null || username.isBlank() || email == null || email.isBlank() || password == null || password.isBlank()){
            log.warn("Registration failed: missing required fields");
            throw new com.example.identity_service.exception.BadRequestException("username_email_password_required");
        }

        if (loginRepository.findByUsername(username).isPresent()){
            log.warn("Registration failed: username taken username={}", username);
            throw new com.example.identity_service.exception.ConflictException("username_taken");
        }
        if (loginRepository.findByEmail(email).isPresent()){
            log.warn("Registration failed: email taken email={}", email);
            throw new com.example.identity_service.exception.ConflictException("email_taken");
        }

        String hash = passwordEncoder.encode(password);
        OffsetDateTime now = OffsetDateTime.now();
        User user = new User(username, email, hash, now);
        User saved = loginRepository.save(user);

        log.info("Registration success: userId={} username={}", saved.getId(), saved.getUsername());
        String token = jwtService.generateToken(saved.getId(), saved.getUsername());
        return token;
    }

    /**
     * Validate a plaintext API key and issue a short-lived JWT scoped for data access.
     * Returns a JWT string when successful, or throws Unauthorized/NotFound.
     */
    public String tokenForApiKey(String plaintextKey){
        if (plaintextKey == null || plaintextKey.isBlank()) {
            log.warn("API key token exchange failed: missing key");
            throw new com.example.identity_service.exception.BadRequestException("api_key_required");
        }

        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(plaintextKey.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) sb.append(String.format("%02x", b));
            String keyHash = sb.toString();

            java.util.Optional<com.example.identity_service.entity.ApiKey> akOpt = apiKeyRepository.findByKeyHash(keyHash);
            if (akOpt.isEmpty()) {
                log.warn("API key token exchange failed: key not found");
                throw new com.example.identity_service.exception.UnauthorizedException("invalid_api_key");
            }
            com.example.identity_service.entity.ApiKey ak = akOpt.get();
            if (!"active".equalsIgnoreCase(ak.getStatus())) {
                log.warn("API key token exchange failed: key inactive teamId={} prefix={}", ak.getTeamId(), ak.getKeyPrefix());
                throw new com.example.identity_service.exception.ForbiddenException("api_key_inactive");
            }

            // update last used timestamp
            ak.setLastUsedAt(java.time.OffsetDateTime.now());
            apiKeyRepository.save(ak);

            // Issue a JWT whose subject is the team id and includes key_prefix
            java.util.UUID teamId = ak.getTeamId();
            log.info("API key token exchange success: teamId={} prefix={}", teamId, ak.getKeyPrefix());
            String token = jwtService.generateTokenForApiKey(teamId, ak.getKeyPrefix(), apiKeyTokenTtlMs);
            return token;
        } catch (Exception ex) {
            throw new com.example.identity_service.exception.RegistrationFailedException();
        }
    }

    /** Return current user profile info. */
    public java.util.Map<String, Object> getMe(UUID userId) {
        Optional<User> userOpt = loginRepository.findById(userId);
        if (userOpt.isEmpty()) throw new com.example.identity_service.exception.NotFoundException("user_not_found");
        User user = userOpt.get();
        return java.util.Map.of(
            "id", user.getId().toString(),
            "username", user.getUsername(),
            "email", user.getEmail(),
            "createdAt", user.getCreatedAt().toString()
        );
    }

    /** Update profile fields (username, email, password). Returns a fresh JWT. */
    @org.springframework.transaction.annotation.Transactional
    public String updateProfile(UUID userId, com.example.identity_service.dto.UpdateProfileDto dto) {
        Optional<User> userOpt = loginRepository.findById(userId);
        if (userOpt.isEmpty()) throw new com.example.identity_service.exception.NotFoundException("user_not_found");
        User user = userOpt.get();

        if (dto.getUsername() != null && !dto.getUsername().isBlank()
                && !dto.getUsername().equals(user.getUsername())) {
            if (loginRepository.findByUsername(dto.getUsername()).isPresent()) {
                throw new com.example.identity_service.exception.ConflictException("username_taken");
            }
            user.setUsername(dto.getUsername());
        }

        if (dto.getEmail() != null && !dto.getEmail().isBlank()
                && !dto.getEmail().equals(user.getEmail())) {
            if (loginRepository.findByEmail(dto.getEmail()).isPresent()) {
                throw new com.example.identity_service.exception.ConflictException("email_taken");
            }
            user.setEmail(dto.getEmail());
        }

        if (dto.getNewPassword() != null && !dto.getNewPassword().isBlank()) {
            if (dto.getCurrentPassword() == null || dto.getCurrentPassword().isBlank()) {
                throw new com.example.identity_service.exception.BadRequestException("current_password_required");
            }
            if (!passwordEncoder.matches(dto.getCurrentPassword(), user.getPasswordHash())) {
                throw new com.example.identity_service.exception.UnauthorizedException("invalid_current_password");
            }
            user.setPasswordHash(passwordEncoder.encode(dto.getNewPassword()));
        }

        loginRepository.save(user);
        log.info("Profile updated: userId={}", userId);
        return jwtService.generateToken(user.getId(), user.getUsername());
    }
}
