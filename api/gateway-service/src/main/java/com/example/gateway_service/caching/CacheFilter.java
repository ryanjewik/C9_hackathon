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
import org.springframework.data.domain.Range;

import java.time.Duration;
import java.util.Base64;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 40)
public class CacheFilter implements GlobalFilter {

    private final ReactiveStringRedisTemplate redis;
    private final long ttlSeconds;
    // index and limit for eviction
    private static final String INDEX_KEY = "cache:index";
    private static final int CACHE_LIMIT = 50;
    // cache payload format: status|contentType|bodyBase64

    public CacheFilter(ReactiveStringRedisTemplate redis,
                       @Value("${cache.ttl-seconds:1800}") long ttlSeconds) {
        this.redis = redis;
        this.ttlSeconds = ttlSeconds;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        // Only cache safe idempotent GETs
        ServerHttpRequest request = exchange.getRequest();
        if (request.getMethod() == null || !"GET".equalsIgnoreCase(request.getMethod().name())) {
            return chain.filter(exchange);
        }
        String method = request.getMethod().name();

        //make the key
        String key = "cache: " + method + ":" + request.getPath().value();
        String parameters = request.getURI().getRawQuery();
        if (parameters != null && !(parameters.isEmpty())){
            key += "?" + parameters;
        }
        final String cacheKey = key;

        return redis.opsForValue().get(cacheKey)/*response looks like '200|application/json|eypcZCI6MX0'*/
        .flatMap(cached->/* this is the variable we are saving the cache "get" to*/{
            //we use .flatMap because .get returns a Mono<String> type, so we use flatMap to work with the String
            try {
                if (cached == null || cached.isEmpty()){
                    return Mono.empty();
                }
                String status = null;
                String contentType = "application/octet-stream";
                String bodyB64 = "";
                if (!(cached.contains("|"))){ //ensure it's parsable for us
                    return Mono.empty();
                }
                String[] parts = cached.split("\\|", 3); //serialize
                if (parts.length > 0){
                    status = parts[0];
                }
                if (parts.length > 1) {
                    contentType = parts[1];
                }
                if (parts.length > 2) {
                    bodyB64 = parts[2];
                }
                byte[] body = new byte[0];
                if (bodyB64 != null && !bodyB64.isEmpty()) {
                    body = Base64.getDecoder().decode(bodyB64);
                }
                // populate the response with updated status code, body, and headers
                ServerHttpResponse response = exchange.getResponse();
                int statusCode = 200;
                if (status != null && !status.isEmpty()) {
                    try {
                        statusCode = Integer.parseInt(status.trim());
                    } catch (NumberFormatException ex) {
                        statusCode = 200;
                    }
                }
                response.setStatusCode(HttpStatus.valueOf(statusCode));
                response.getHeaders().set(HttpHeaders.CONTENT_TYPE, contentType);
                return response.writeWith(Mono.just(response.bufferFactory().wrap(body)));
            } catch (Exception e){
                return Mono.empty();
            }
        }).switchIfEmpty(Mono.defer(()-> { //if cache miss
            ServerHttpResponse original = exchange.getResponse(); //read buffer
            ServerHttpResponseDecorator decorated = new ServerHttpResponseDecorator(original){
                    @Override
                    @SuppressWarnings("unchecked")
                    public Mono<Void> writeWith(org.reactivestreams.Publisher<? extends DataBuffer> body) {
                        if (body instanceof Flux) {
                            Flux<? extends DataBuffer> fluxBody = Flux.from(body);
                            return DataBufferUtils.join(fluxBody).flatMap(dataBuffer -> {
                            byte[] content = new byte[dataBuffer.readableByteCount()];
                            dataBuffer.read(content);
                            DataBufferUtils.release(dataBuffer);

                            //only cache successful 2xx responses
                            var status = this.getStatusCode();
                            if (status != null && status.is2xxSuccessful()){
                                try {
                                    int statusCode = status.value();
                                    String contentType = this.getHeaders().getFirst(HttpHeaders.CONTENT_TYPE);
                                    if (contentType == null){
                                        contentType = "application/octet-stream";
                                    }
                                    String payload = statusCode + "|" + contentType + "|" + Base64.getEncoder().encodeToString(content);
                                    long now = System.currentTimeMillis();
                                    // set value, add to index, then evict oldest if over limit (runs async)
                                    redis.opsForValue().set(cacheKey, payload, Duration.ofSeconds(ttlSeconds))
                                        .then(redis.opsForZSet().add(INDEX_KEY, cacheKey, (double) now))
                                        .then(redis.opsForZSet().size(INDEX_KEY))
                                        .flatMap(size -> {
                                            if (size <= CACHE_LIMIT) return Mono.<Void>empty();
                                            long toRemove = size - CACHE_LIMIT;
                                            return redis.opsForZSet().range(INDEX_KEY, Range.closed(0L, toRemove - 1L)).collectList()
                                                .flatMap(keys -> {
                                                    if (keys == null || keys.isEmpty()) return Mono.<Void>empty();
                                                    String[] arr = keys.toArray(new String[0]);
                                                    return redis.opsForZSet().remove(INDEX_KEY, (Object[]) arr)
                                                        .then(Flux.fromArray(arr)
                                                            .flatMap(k -> redis.delete(k))
                                                            .then());
                                                });
                                        })
                                        .subscribe();
                                } catch (Exception ex){
                                    //ignore cache errors
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
