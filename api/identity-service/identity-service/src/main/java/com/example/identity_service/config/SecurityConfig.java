package com.example.identity_service.config;

import javax.crypto.spec.SecretKeySpec;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.core.Ordered;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.oauth2.jwt.ReactiveJwtDecoder;
import org.springframework.security.oauth2.jwt.NimbusReactiveJwtDecoder;
import java.security.interfaces.RSAPublicKey;
import com.example.identity_service.service.KeyService;

@Configuration
@EnableWebFluxSecurity
@Order(Ordered.HIGHEST_PRECEDENCE)
public class SecurityConfig {
    @Bean
    @Order(Ordered.HIGHEST_PRECEDENCE)
    public SecurityWebFilterChain springSecurityFilterChain(ServerHttpSecurity http) {
        http
            .csrf(csrf -> csrf.disable())
            .authorizeExchange(exchanges -> exchanges
                .pathMatchers("/teamadmin/**").authenticated()
                .pathMatchers("/auth/login", "/auth/register", "/actuator/**").permitAll()
                .anyExchange().permitAll()
            )
            .oauth2ResourceServer(oauth2 -> oauth2.jwt(jwt -> {
                // No-op customizer: ReactiveJwtDecoder bean will be picked up from context
            }));

        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public ReactiveJwtDecoder reactiveJwtDecoder(@Value("${jwt.secret:}") String secret, KeyService keyService) {
        // If KeyService provides an RSA public key, use it to validate RS256 tokens.
        try {
            RSAPublicKey pub = keyService != null ? keyService.getPublicKey() : null;
            if (pub != null) {
                return NimbusReactiveJwtDecoder.withPublicKey(pub).build();
            }
        } catch (Exception e) {
            // fall through to HMAC fallback
        }

        // Fallback to HMAC-based JWTs using the configured secret
        if (secret == null || secret.isEmpty()) {
            throw new IllegalArgumentException("jwt.secret must be set when no RSA key is available");
        }
        SecretKeySpec key = new SecretKeySpec(secret.getBytes(), "HmacSHA256");
        return NimbusReactiveJwtDecoder.withSecretKey(key).build();
    }
}
