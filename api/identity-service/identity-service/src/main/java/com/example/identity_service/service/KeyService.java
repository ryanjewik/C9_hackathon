package com.example.identity_service.service;

import com.nimbusds.jose.jwk.RSAKey;
import com.nimbusds.jose.jwk.JWKSet;
import com.nimbusds.jose.jwk.KeyUse;
import com.nimbusds.jose.JWSAlgorithm;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.interfaces.RSAPrivateKey;
import java.security.interfaces.RSAPrivateCrtKey;
import java.security.interfaces.RSAPublicKey;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.RSAPublicKeySpec;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

@Component
public class KeyService {

    private RSAPrivateKey privateKey;
    private RSAPublicKey publicKey;
    private RSAKey jwk;

    @PostConstruct
    public void init() throws Exception {
        // Prefer mounted secret at /run/secrets/jwt_private.pem if present
        Path p = Path.of(System.getenv().getOrDefault("JWT_PRIVATE_KEY_PATH", "/run/secrets/jwt_private.pem"));
        if (Files.exists(p)) {
            try {
                String pem = Files.readString(p, StandardCharsets.UTF_8);
                // Strip PEM headers/footers
                pem = pem.replaceAll("-----BEGIN (.*)-----", "");
                pem = pem.replaceAll("-----END (.*)-----", "");
                pem = pem.replaceAll("\\s+", "");
                byte[] der = Base64.getDecoder().decode(pem);
                PKCS8EncodedKeySpec spec = new PKCS8EncodedKeySpec(der);
                KeyFactory kf = KeyFactory.getInstance("RSA");
                RSAPrivateKey priv = (RSAPrivateKey) kf.generatePrivate(spec);
                this.privateKey = priv;

                // Try to derive public key from private (works for RSA CRT keys)
                if (priv instanceof RSAPrivateCrtKey) {
                    RSAPrivateCrtKey crt = (RSAPrivateCrtKey) priv;
                    RSAPublicKeySpec pubSpec = new RSAPublicKeySpec(crt.getModulus(), crt.getPublicExponent());
                    this.publicKey = (RSAPublicKey) kf.generatePublic(pubSpec);
                }
            } catch (Exception e) {
                // Fall back to generated keypair on error
                this.privateKey = null;
                this.publicKey = null;
            }
        }

        if (this.privateKey == null || this.publicKey == null) {
            KeyPairGenerator kpg = KeyPairGenerator.getInstance("RSA");
            kpg.initialize(2048);
            KeyPair kp = kpg.generateKeyPair();
            this.privateKey = (RSAPrivateKey) kp.getPrivate();
            this.publicKey = (RSAPublicKey) kp.getPublic();
        }

        this.jwk = new RSAKey.Builder(this.publicKey)
                .keyUse(KeyUse.SIGNATURE)
                .algorithm(JWSAlgorithm.RS256)
                .keyID(System.getenv().getOrDefault("JWT_KEY_ID", "dev-key"))
                .build();
    }

    public RSAPrivateKey getPrivateKey() {
        return privateKey;
    }

    public RSAPublicKey getPublicKey() {
        return publicKey;
    }

    public JWKSet getJwkSet() {
        return new JWKSet(jwk);
    }
}
