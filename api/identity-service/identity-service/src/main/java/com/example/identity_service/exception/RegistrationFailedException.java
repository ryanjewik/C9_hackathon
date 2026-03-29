package com.example.identity_service.exception;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND)
public class RegistrationFailedException extends RuntimeException{
    public RegistrationFailedException() {
        super("Registration failed! Username/email was already taken, try logging in or using different credentials.");
    }
}
