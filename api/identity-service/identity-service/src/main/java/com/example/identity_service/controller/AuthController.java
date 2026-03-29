package com.example.identity_service.controller;

import com.example.identity_service.dto.LoginDto;
import com.example.identity_service.dto.RegisterDto;
import com.example.identity_service.dto.ApikeyDto;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import com.example.identity_service.service.AuthService;
import java.util.Map;

import java.util.Optional;


@RestController
@RequestMapping("auth")
public class AuthController {
    
    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginDto credentials){
        String username = credentials.getUsername();
        String password = credentials.getPassword();
        Optional<String> tokenOpt = authService.authenticate(username, password);
        if (tokenOpt.isEmpty()){
            return ResponseEntity.status(401).body(Map.of("error", "invalid_credentials"));
        }

        String token = tokenOpt.get();
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

        Optional<String> tokenOpt = authService.register(username, email, password);
        if (tokenOpt.isEmpty()){
            return ResponseEntity.status(409).body(Map.of("error", "username_or_email_taken_or_invalid"));
        }

        String token = tokenOpt.get();
        return ResponseEntity.status(201).body(Map.of(
            "access_token", token,
            "token_type", "Bearer"
        ));
    }

    // @PostMapping("apikey")
    // public ResponseEntity<?> apikey(@RequestBody ApikeyDto body){
    //     return ResponseEntity.status(201).body(Map.of(
    //         "api_key", "blank"
    //     ));
    // }
    
}
