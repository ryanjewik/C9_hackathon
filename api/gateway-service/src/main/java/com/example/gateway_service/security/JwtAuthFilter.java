package com.example.gateway_service.security;

import org.springframework.stereotype.Component;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.cloud.gateway.filter.NettyWriteResponseFilter;
import org.springframework.http.HttpHeaders;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.http.HttpStatus;
import org.springframework.beans.factory.annotation.Value;

import reactor.core.publisher.Mono;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Claims;

import java.security.Key;
import io.jsonwebtoken.security.Keys;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class JwtAuthFilter implements GlobalFilter {

    private final Key key;

    public JwtAuthFilter(@Value("${jwt.secret:change_me_replace_with_env_secret}") String secret) {
        this.key = Keys.hmacShaKeyFor(secret.getBytes());
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, org.springframework.cloud.gateway.filter.GatewayFilterChain chain) {
        String path = exchange.getRequest().getPath().value();

        // Only enforce auth for /api/** routes (forwarding to data-service)
        if (!path.startsWith("/api/")) {
            return chain.filter(exchange);
        }

        HttpHeaders headers = exchange.getRequest().getHeaders();
        String auth = headers.getFirst(HttpHeaders.AUTHORIZATION);
        if (auth == null || !auth.startsWith("Bearer ")) {
            ServerHttpResponse resp = exchange.getResponse();
            resp.setStatusCode(HttpStatus.UNAUTHORIZED);
            return resp.setComplete();
        }

        String token = auth.substring("Bearer ".length());
        Claims claims;
        try {
            claims = Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(token).getBody();
        } catch (JwtException ex) {
            ServerHttpResponse resp = exchange.getResponse();
            resp.setStatusCode(HttpStatus.UNAUTHORIZED);
            return resp.setComplete();
        }

        ServerHttpRequest req = exchange.getRequest().mutate()
                .headers(h -> {
                    // Remove original Authorization header before forwarding to internal services
                    h.remove(HttpHeaders.AUTHORIZATION);
                    // Map subject to explicit downstream header names
                    String subject = claims.getSubject();
                    Object apikey = claims.get("apikey");
                    if (apikey != null && Boolean.TRUE.equals(Boolean.valueOf(apikey.toString()))) {
                        // API-key-derived token: subject is a team id
                        if (subject != null) h.add("X-Team-Id", subject);
                        h.add("X-Auth-Method", "apikey-token");
                        Object kp = claims.get("key_prefix");
                        if (kp != null) h.add("X-Key-Prefix", kp.toString());
                    } else {
                        // User JWT: subject is a user id
                        if (subject != null) h.add("X-User-Id", subject);
                        h.add("X-Auth-Method", "jwt");
                    }

                    Object username = claims.get("username");
                    if (username != null) h.add("X-Username", username.toString());
                })
                .build();

        ServerWebExchange mutated = exchange.mutate().request(req).build();
        return chain.filter(mutated);
    }
}
