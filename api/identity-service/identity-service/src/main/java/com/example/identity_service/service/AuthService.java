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
    public Optional<String> authenticate(String username, String password){
        Optional<User> userOpt = loginRepository.findByUsername(username);
        if (userOpt.isEmpty()){
            return Optional.empty();
        }
        User user = userOpt.get();
        if (!passwordEncoder.matches(password, user.getPasswordHash())){
            return Optional.empty();
        }

        String token = jwtService.generateToken(user.getId(), user.getUsername());
        return Optional.of(token);
    }

    /**
     * Register a new user. Returns Optional JWT when successful.
     * If username or email already exists, returns Optional.empty().
     */
    public Optional<String> register(String username, String email, String password){
        if (username == null || username.isBlank() || email == null || email.isBlank() || password == null || password.isBlank()){
            return Optional.empty();
        }

        if (loginRepository.findByUsername(username).isPresent()){
            return Optional.empty();
        }
        if (loginRepository.findByEmail(email).isPresent()){
            return Optional.empty();
        }

        String hash = passwordEncoder.encode(password);
        OffsetDateTime now = OffsetDateTime.now();
        User user = new User(username, email, hash, now);
        User saved = loginRepository.save(user);

        String token = jwtService.generateToken(saved.getId(), saved.getUsername());
        return Optional.of(token);
    }
}
