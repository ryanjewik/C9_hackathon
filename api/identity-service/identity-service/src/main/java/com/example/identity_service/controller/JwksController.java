package com.example.identity_service.controller;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.beans.factory.annotation.Autowired;
import com.nimbusds.jose.jwk.JWKSet;
import com.example.identity_service.service.KeyService;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class JwksController {

    private final KeyService keyService;

    @Autowired
    public JwksController(KeyService keyService) {
        this.keyService = keyService;
    }

    @GetMapping(path = "/.well-known/jwks.json", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Map<String, Object>> jwks() {
        JWKSet set = keyService.getJwkSet();
        return ResponseEntity.ok(set.toJSONObject());
    }
}
