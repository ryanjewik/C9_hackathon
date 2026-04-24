package com.example.identity_service.controller;

import com.example.identity_service.dto.LoginDto;
import com.example.identity_service.dto.RegisterDto;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import com.example.identity_service.service.AuthService;
import com.example.identity_service.dto.ApiKeyTokenRequestDto;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;


@RestController
@RequestMapping("auth")
public class AuthController {
    
    private final AuthService authService;
    private static final Logger log = LoggerFactory.getLogger(AuthController.class);
    private final long apiKeyTokenTtlMs;

    public AuthController(AuthService authService, @org.springframework.beans.factory.annotation.Value("${jwt.apikey-expiration-ms:900000}") long apiKeyTokenTtlMs) {
        this.authService = authService;
        this.apiKeyTokenTtlMs = apiKeyTokenTtlMs;
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginDto credentials){
        String username = credentials.getUsername();
        String password = credentials.getPassword();
        log.info("Login attempt: username={}", username);
        String token = authService.authenticate(username, password);
        log.info("Login success: username={}", username);
        return ResponseEntity.ok(Map.of(
            "access_token", token,
            "token_type", "Bearer"
        ));
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterDto body){
        String username = body.getUsername();
        String email = body.getEmail();
        String password = body.getPassword();
        log.info("Registration attempt: username={} email={}", username, email);
        String token = authService.register(username, email, password);
        log.info("Registration success: username={}", username);
        return ResponseEntity.status(201).body(Map.of(
            "access_token", token,
            "token_type", "Bearer"
        ));
    }

    @PostMapping("/token")
    public ResponseEntity<?> tokenByApiKey(@RequestBody ApiKeyTokenRequestDto req){
        // Accept plaintext API key in the request body as { "key": "<plaintext>" }
        String key = req.getKey();
        log.info("API key token exchange attempt");
        String token = authService.tokenForApiKey(key);
        log.info("API key token exchange success");
        return ResponseEntity.ok(Map.of(
            "access_token", token,
            "token_type", "Bearer",
            "expires_in", Long.valueOf(apiKeyTokenTtlMs / 1000)
        ));
    }
}
