package com.griddb.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.sql.SQLException;
import java.util.Map;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<?> handleBadRequest(IllegalArgumentException e) {
        return ResponseEntity.badRequest()
                .body(Map.of("error", e.getMessage(), "code", "BAD_REQUEST"));
    }

    @ExceptionHandler(SQLException.class)
    public ResponseEntity<?> handleSql(SQLException e) {
        boolean isAuth = e.getMessage() != null &&
                e.getMessage().toLowerCase().contains("access denied");
        int status = isAuth ? 401 : 500;
        return ResponseEntity.status(status).body(Map.of(
                "error", e.getMessage() != null ? e.getMessage() : "SQL error",
                "code", "SQL_ERROR",
                "sqlState", e.getSQLState() != null ? e.getSQLState() : ""
        ));
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<?> handleMaxSize(MaxUploadSizeExceededException e) {
        return ResponseEntity.status(413)
                .body(Map.of("error", "File too large. Max upload size is 500 MB.", "code", "FILE_TOO_LARGE"));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<?> handleGeneral(Exception e) {
        return ResponseEntity.status(500).body(Map.of(
                "error", e.getMessage() != null ? e.getMessage() : "Internal server error",
                "code", "INTERNAL_ERROR"
        ));
    }
}
