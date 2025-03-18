package com.yongu2000.mbtitranslator.controller;

import com.yongu2000.mbtitranslator.dto.TranslateRequest;
import com.yongu2000.mbtitranslator.dto.TranslateResponse;
import com.yongu2000.mbtitranslator.service.TranslateService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/translate")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:3000")  // Next.js 프론트엔드 주소
public class TranslateController {
    private final TranslateService translateService;

    @PostMapping
    public ResponseEntity<TranslateResponse> translate(@RequestBody TranslateRequest request) {
        TranslateResponse response = translateService.translate(
            request.getSourceMbti(),
            request.getTargetMbti(),
            request.getText()
        );
        return ResponseEntity.ok(response);
    }
}
