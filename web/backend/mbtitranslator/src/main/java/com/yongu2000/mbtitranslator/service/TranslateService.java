package com.yongu2000.mbtitranslator.service;

import com.yongu2000.mbtitranslator.dto.TranslateRequest;
import com.yongu2000.mbtitranslator.dto.TranslateResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
@Slf4j
public class TranslateService {
    private final String MODEL_SERVER_URL = "http://localhost:8000";
    private final RestTemplate restTemplate;

    public TranslateService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public String translate(String sourceMbti, String targetMbti, String text) {
        try {
            TranslateRequest request = new TranslateRequest();
            request.setSourceMbti(sourceMbti);
            request.setTargetMbti(targetMbti);
            request.setText(text);

            ResponseEntity<TranslateResponse> response = restTemplate.postForEntity(
                MODEL_SERVER_URL + "/translate",
                request,
                TranslateResponse.class
            );

            return response.getBody().getTranslatedText();
        } catch (Exception e) {
            log.error("번역 요청 실패", e);
            throw new RuntimeException("번역 처리 중 오류 발생", e);
        }
    }
}
