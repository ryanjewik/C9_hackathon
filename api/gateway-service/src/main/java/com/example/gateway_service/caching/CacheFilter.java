package com.example.gateway_service.caching;

import org.springframework.stereotype.Component;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Flux;

import org.springframework.data.redis.core.ReactiveStringRedisTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.server.reactive.ServerHttpResponseDecorator;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpHeaders;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;

import java.time.Duration;
import java.util.Base64;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 40)
public class CacheFilter implements GlobalFilter {

    private final ReactiveStringRedisTemplate redis;
    private final long ttlSeconds;
    // cache payload format: status|contentType|bodyBase64

    public CacheFilter(ReactiveStringRedisTemplate redis,
                       @Value("${cache.ttl-seconds:1800}") long ttlSeconds) {
        this.redis = redis;
        this.ttlSeconds = ttlSeconds;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        // Only cache safe idempotent GETs
        ServerHttpRequest req = exchange.getRequest();
        String method = req.getMethod() != null ? req.getMethod().name() : "";
        if (!"GET".equalsIgnoreCase(method)) {
            return chain.filter(exchange);
        }

        // derive cache key (global across teams): method + path + querystring
        String key = "cache:" + method + ":" + req.getPath().value();
        String qs = req.getURI().getRawQuery();
        if (qs != null && !qs.isEmpty()) key += "?" + qs;
        final String cacheKey = key;

        // Try cache hit
        return redis.opsForValue().get(cacheKey)
            .flatMap(cached -> {
                try {
                    if (cached == null || cached.isEmpty()) return Mono.empty();
                    String statusStr = null;
                    String contentType = "application/octet-stream";
                    String bodyB64 = "";

                    // support simple pipe-separated format: status|contentType|bodyBase64
                    if (cached.contains("|")) {
                        String[] parts = cached.split("\\|", 3);
                        statusStr = parts.length > 0 ? parts[0] : null;
                        contentType = parts.length > 1 ? parts[1] : contentType;
                        bodyB64 = parts.length > 2 ? parts[2] : "";
                    } else {
                        // naive JSON fallback: look for "status":NN and "contentType":"..." and "body":"..."
                        int si = cached.indexOf("\"status\"");
                        if (si >= 0) {
                            int col = cached.indexOf(':', si);
                            if (col >= 0) {
                                int comma = cached.indexOf(',', col);
                                int end = comma > 0 ? comma : cached.indexOf('}', col);
                                if (end > col) statusStr = cached.substring(col + 1, end).trim().replaceAll("[\" ]", "");
                            }
                        }
                        int ci = cached.indexOf("\"contentType\"");
                        if (ci >= 0) {
                            int col = cached.indexOf(':', ci);
                            int start = cached.indexOf('"', col + 1);
                            int end = start >= 0 ? cached.indexOf('"', start + 1) : -1;
                            if (start >= 0 && end > start) contentType = cached.substring(start + 1, end);
                        }
                        int bi = cached.indexOf("\"body\"");
                        if (bi >= 0) {
                            int col = cached.indexOf(':', bi);
                            int start = cached.indexOf('"', col + 1);
                            int end = start >= 0 ? cached.indexOf('"', start + 1) : -1;
                            if (start >= 0 && end > start) bodyB64 = cached.substring(start + 1, end);
                        }
                    }

                    int status = 200;
                    if (statusStr != null && !statusStr.isEmpty()) {
                        try { status = Integer.parseInt(statusStr); } catch (NumberFormatException ex) { status = 200; }
                    }

                    byte[] body = bodyB64 != null && !bodyB64.isEmpty() ? Base64.getDecoder().decode(bodyB64) : new byte[0];

                    ServerHttpResponse resp = exchange.getResponse();
                    resp.setStatusCode(HttpStatus.valueOf(status));
                    resp.getHeaders().set(HttpHeaders.CONTENT_TYPE, contentType);
                    return resp.writeWith(Mono.just(resp.bufferFactory().wrap(body)));
                } catch (Exception e) {
                    // if cache is corrupt, proceed to fetch fresh
                    return Mono.empty();
                }
            })
            .switchIfEmpty(Mono.defer(() -> {
                // cache miss: decorate the response to capture body
                ServerHttpResponse original = exchange.getResponse();
                ServerHttpResponseDecorator decorated = new ServerHttpResponseDecorator(original) {
                    @Override
                    @SuppressWarnings("unchecked")
                    public Mono<Void> writeWith(org.reactivestreams.Publisher<? extends DataBuffer> body) {
                        if (body instanceof Flux) {
                            Flux<? extends DataBuffer> fluxBody = Flux.from(body);
                            return DataBufferUtils.join(fluxBody)
                                .flatMap(dataBuffer -> {
                                    byte[] content = new byte[dataBuffer.readableByteCount()];
                                    dataBuffer.read(content);
                                    DataBufferUtils.release(dataBuffer);

                                    // only cache successful 2xx responses
                                    var statusCode = this.getStatusCode();
                                    if (statusCode != null && statusCode.is2xxSuccessful()) {
                                        try {
                                            int code = statusCode.value();
                                            String ct = this.getHeaders().getFirst(HttpHeaders.CONTENT_TYPE);
                                            String payload = code + "|" + (ct != null ? ct : "application/octet-stream") + "|" + Base64.getEncoder().encodeToString(content);
                                            // store asynchronously but continue writing immediately
                                            redis.opsForValue().set(cacheKey, payload, Duration.ofSeconds(ttlSeconds)).subscribe();
                                        } catch (Exception ex) {
                                            // ignore cache errors
                                        }
                                    }

                                    DataBuffer buffer = this.bufferFactory().wrap(content);
                                    return super.writeWith(Mono.just(buffer));
                                });
                        }
                        return super.writeWith(body);
                    }
                };

                ServerWebExchange mutated = exchange.mutate().response(decorated).build();
                return chain.filter(mutated);
            }));
    }
}
