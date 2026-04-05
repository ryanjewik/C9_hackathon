package com.example.identity_service.controller;

import com.example.identity_service.dto.LoginDto;
import com.example.identity_service.dto.RegisterDto;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import com.example.identity_service.service.AuthService;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;


@RestController
@RequestMapping("auth")
public class AuthController {
    
    private final AuthService authService;
    private static final Logger log = LoggerFactory.getLogger(AuthController.class);

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginDto credentials){
        String username = credentials.getUsername();
        String password = credentials.getPassword();
        String token = authService.authenticate(username, password);
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
        String token = authService.register(username, email, password);
        return ResponseEntity.status(201).body(Map.of(
            "access_token", token,
            "token_type", "Bearer"
        ));
    }
}
