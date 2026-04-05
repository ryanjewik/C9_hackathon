package com.example.identity_service.exception;

public class BadRequestException extends RuntimeException {
    public BadRequestException(String message) { super(message); }
}
