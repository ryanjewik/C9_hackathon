package com.example.identity_service.service;
import org.springframework.stereotype.Service;
import com.example.identity_service.entity.User;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.Date;
import java.util.Optional;
import com.example.identity_service.repository.AuthRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import java.util.UUID;

@Service
public class AuthService {
    private final AuthRepository loginRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthService(AuthRepository loginRepository, PasswordEncoder passwordEncoder, JwtService jwtService){
        this.loginRepository = loginRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
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
}
