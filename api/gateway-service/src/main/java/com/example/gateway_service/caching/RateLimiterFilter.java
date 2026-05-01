package com.example.gateway_service.caching;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.beans.factory.annotation.Value;
import reactor.core.publisher.Mono;
import org.springframework.data.redis.core.ReactiveStringRedisTemplate;
import org.springframework.data.redis.core.script.RedisScript;
import org.springframework.core.io.ClassPathResource;
import org.springframework.util.StreamUtils;

import java.time.Instant;
import java.util.Collections;
import java.nio.charset.StandardCharsets;
import java.io.IOException;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 20)
public class RateLimiterFilter implements GlobalFilter {
    private static final Logger log = LoggerFactory.getLogger(RateLimiterFilter.class);
    private final ReactiveStringRedisTemplate redis;
    private long windowMillis;
    private long maxRequests;
    private final RedisScript<Long> script;

    public RateLimiterFilter(ReactiveStringRedisTemplate redis,
        @Value("${ratelimit.window-seconds:60}") long windowSeconds,
        @Value("${ratelimit.requests-per-window:100}") long maxRequests) {
            this.redis = redis;
            this.windowMillis = windowSeconds * 1000L;
            this.maxRequests = maxRequests;

            ClassPathResource res = new ClassPathResource("scripts/SlidingWindow.lua");
            try {
                String lua = StreamUtils.copyToString(res.getInputStream(), StandardCharsets.UTF_8);
                this.script = RedisScript.of(lua, Long.class);
            } catch(IOException e){
                throw new IllegalStateException("Unable to load sliding window lua script", e);
            }
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        // Public dashboard endpoints are exempt from rate limiting
        if (exchange.getRequest().getPath().value().startsWith("/dashboard/")) {
            return chain.filter(exchange);
        }

        String _teamId = exchange.getRequest().getHeaders().getFirst("X-Team-Id");
        if (_teamId == null || _teamId.isEmpty()) {
            if (exchange.getRequest().getRemoteAddress() != null && exchange.getRequest().getRemoteAddress().getAddress() != null) {
                _teamId = exchange.getRequest().getRemoteAddress().getAddress().getHostAddress();
            }
        }
        final String teamId = _teamId;

        String key = "ratelimit:team:" + teamId;
        long now = Instant.now().toEpochMilli();

        return redis.execute(script, Collections.singletonList(key), String.valueOf(now), String.valueOf(windowMillis), String.valueOf(maxRequests))
            .next()
            .flatMap(count -> {
                long current = count != null ? count.longValue() : 0L;
                if (current > maxRequests) {
                    log.warn("Rate limit exceeded: id={} count={} limit={}", teamId, current, maxRequests);
                    ServerHttpResponse response = exchange.getResponse();
                    response.setStatusCode(HttpStatus.TOO_MANY_REQUESTS);
                    response.getHeaders().add("Retry-After", String.valueOf(windowMillis / 1000L));
                    response.getHeaders().add("X-RateLimit-Limit", String.valueOf(maxRequests));
                    response.getHeaders().add("X-RateLimit-Remaining", "0");
                    return response.setComplete();
                }

                log.debug("Rate limit OK: id={} count={} remaining={}", teamId, current, Math.max(0, maxRequests - current));
                exchange.getResponse().getHeaders().add("X-RateLimit-Limit", String.valueOf(maxRequests));
                exchange.getResponse().getHeaders().add("X-RateLimit-Remaining", String.valueOf(Math.max(0, maxRequests - current)));
                return chain.filter(exchange);
            });
    }
            
}