package com.example.identity_service.controller;

import com.example.identity_service.dto.CredentialsDto;
import com.example.identity_service.dto.RegisterDto;
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
    //initialize service
    private final AuthService loginService;

    public AuthController(AuthService loginService) {
        this.loginService = loginService;
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody CredentialsDto credentials){
        String username = credentials.getUsername();
        String password = credentials.getPassword();
        Optional<String> tokenOpt = loginService.authenticate(username, password);
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

        Optional<String> tokenOpt = loginService.register(username, email, password);
        if (tokenOpt.isEmpty()){
            return ResponseEntity.status(409).body(Map.of("error", "username_or_email_taken_or_invalid"));
        }

        String token = tokenOpt.get();
        return ResponseEntity.status(201).body(Map.of(
            "access_token", token,
            "token_type", "Bearer"
        ));
    }
    //create session token?
}
