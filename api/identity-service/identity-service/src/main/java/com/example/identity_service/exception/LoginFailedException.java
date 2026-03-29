package com.example.identity_service.exception;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND)
public class LoginFailedException extends RuntimeException{
    public LoginFailedException() {
        super("Login failed! Please try another password/username.");
    }
}
