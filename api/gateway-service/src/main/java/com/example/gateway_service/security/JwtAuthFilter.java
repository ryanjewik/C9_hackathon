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
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;

import java.security.Key;
import io.jsonwebtoken.security.Keys;
import java.util.Map;
import java.util.Collection;

import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class JwtAuthFilter implements GlobalFilter {

    private static final Logger log = LoggerFactory.getLogger(JwtAuthFilter.class);
    private final Key key;
    private final String expectedIssuer;
    private final String expectedAudience;
    private final JwtDecoder jwtDecoder;

    public JwtAuthFilter(
            @Value("${jwt.secret:change_me_replace_with_env_secret}") String secret,
            @Value("${jwt.issuer:}") String issuer,
            @Value("${jwt.audience:}") String audience,
            @Value("${jwt.jwks-uri:}") String jwksUri) {
        this.key = Keys.hmacShaKeyFor(secret.getBytes());
        this.expectedIssuer = (issuer == null || issuer.isEmpty()) ? null : issuer;
        this.expectedAudience = (audience == null || audience.isEmpty()) ? null : audience;
        this.jwtDecoder = (jwksUri == null || jwksUri.isEmpty()) ? null : NimbusJwtDecoder.withJwkSetUri(jwksUri).build();
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, org.springframework.cloud.gateway.filter.GatewayFilterChain chain) {
        String path = exchange.getRequest().getPath().value();
        boolean isApi = path.startsWith("/api/");
        boolean isIdentity = path.startsWith("/auth/") || path.startsWith("/teamadmin/");

        if (!isApi && !isIdentity) {
            return chain.filter(exchange);
        }

        // Allow unauthenticated access to public identity endpoints such as
        // POST /auth/register and POST /auth/login so clients can register/login
        // without a JWT. Keep other identity endpoints protected.
        if (isIdentity && (path.equals("/auth/register") || path.equals("/auth/login") || path.equals("/auth/token") || path.startsWith("/auth/public/"))) {
            log.debug("Auth bypass: public endpoint {}", path);
            return chain.filter(exchange);
        }

        HttpHeaders headers = exchange.getRequest().getHeaders();
        String auth = headers.getFirst(HttpHeaders.AUTHORIZATION);
        if (auth == null || !auth.startsWith("Bearer ")) {
            log.warn("Missing or invalid Authorization header: method={} path={}", exchange.getRequest().getMethod(), path);
            ServerHttpResponse resp = exchange.getResponse();
            resp.setStatusCode(HttpStatus.UNAUTHORIZED);
            return resp.setComplete();
        }

        String token = auth.substring("Bearer ".length());
        Map<String, Object> claimsMap = null;
        String subject = null;

        // Decode using JWKS (RS256) if configured, otherwise fall back to HMAC secret parsing
        if (this.jwtDecoder != null) {
            try {
                Jwt jwt = this.jwtDecoder.decode(token);
                claimsMap = jwt.getClaims();
                subject = jwt.getSubject();
            } catch (org.springframework.security.oauth2.jwt.JwtException ex) {
                log.warn("JWT decode failed (JWKS): path={} error={}", path, ex.getMessage());
                ServerHttpResponse resp = exchange.getResponse();
                resp.setStatusCode(HttpStatus.UNAUTHORIZED);
                return resp.setComplete();
            }
        } else {
            try {
                // will throw exception if expired or invalid
                Claims jj = Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(token).getBody();
                claimsMap = jj;
                subject = jj.getSubject();
            } catch (io.jsonwebtoken.JwtException ex) {
                log.warn("JWT decode failed (HMAC): path={} error={}", path, ex.getMessage());
                ServerHttpResponse resp = exchange.getResponse();
                resp.setStatusCode(HttpStatus.UNAUTHORIZED);
                return resp.setComplete();
            }
        }

        // Validate issuer and audience if configured
        if (this.expectedIssuer != null) {
            Object issObj = claimsMap.get("iss");
            String iss = (issObj instanceof String) ? (String) issObj : null;
            if (iss == null || !this.expectedIssuer.equals(iss)) {
                log.warn("JWT issuer mismatch: path={} expected={} got={}", path, this.expectedIssuer, iss);
                ServerHttpResponse resp = exchange.getResponse();
                resp.setStatusCode(HttpStatus.UNAUTHORIZED);
                return resp.setComplete();
            }
        }
        if (this.expectedAudience != null) {
            Object audObj = claimsMap.get("aud");
            boolean audOk = false;
            if (audObj instanceof String) {
                audOk = this.expectedAudience.equals(audObj);
            } else if (audObj instanceof Collection) {
                audOk = ((Collection<?>) audObj).contains(this.expectedAudience);
            }
            if (!audOk) {
                log.warn("JWT audience mismatch: path={} expected={}", path, this.expectedAudience);
                ServerHttpResponse resp = exchange.getResponse();
                resp.setStatusCode(HttpStatus.UNAUTHORIZED);
                return resp.setComplete();
            }
        }

        // Create effectively-final references for use inside lambdas below
        final Map<String, Object> finalClaims = claimsMap;
        final String finalSubject = subject;

        // Enforce API-key token for data routes and strip Authorization only for those routes.
        if (isApi) {
            Object apikey = finalClaims.get("apikey");
            if (apikey == null || !Boolean.TRUE.equals(Boolean.valueOf(apikey.toString()))) {
                log.warn("API route requires apikey token: path={} subject={}", path, finalSubject);
                ServerHttpResponse resp = exchange.getResponse();
                resp.setStatusCode(HttpStatus.FORBIDDEN);
                return resp.setComplete();
            }

            ServerHttpRequest req = exchange.getRequest().mutate()
                    .headers(h -> {
                        // Remove original Authorization header before forwarding to internal services
                        h.remove(HttpHeaders.AUTHORIZATION);
                        // API-key-derived token: subject is a team id
                        if (finalSubject != null) h.add("X-Team-Id", finalSubject);
                        h.add("X-Auth-Method", "apikey-token");
                        Object kp = finalClaims.get("key_prefix");
                        if (kp != null) h.add("X-Key-Prefix", kp.toString());
                        Object username = finalClaims.get("username");
                        if (username != null) h.add("X-Username", username.toString());
                    })
                    .build();

            ServerWebExchange mutated = exchange.mutate().request(req).build();
            log.info("API request authorized: path={} teamId={}", path, finalSubject);
            return chain.filter(mutated);
        } else {
            // Identity routes: keep original Authorization header so identity service can introspect/refresh
            ServerHttpRequest req = exchange.getRequest().mutate()
                    .headers(h -> {
                        // Optionally surface user info for convenience
                        if (finalSubject != null) h.add("X-User-Id", finalSubject);
                        h.add("X-Auth-Method", "jwt");
                        Object username = finalClaims.get("username");
                        if (username != null) h.add("X-Username", username.toString());
                    })
                    .build();

            ServerWebExchange mutated = exchange.mutate().request(req).build();
            log.info("Identity request authorized: path={} userId={}", path, finalSubject);
            return chain.filter(mutated);
        }
    }
}
