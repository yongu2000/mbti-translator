package com.yongu2000.mbtitranslator.service;

import com.yongu2000.mbtitranslator.dto.TranslateRequest;
import com.yongu2000.mbtitranslator.dto.TranslateResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
@RequiredArgsConstructor
@Slf4j
public class TranslateService {

    private final RestTemplate restTemplate;

    public TranslateResponse translate(String sourceMbti, String targetMbti, String text) {
        try {
            TranslateRequest request = new TranslateRequest();
            request.setSourceMbti(sourceMbti);
            request.setTargetMbti(targetMbti);
            request.setText(text);
            log.info("source mbti = {}", sourceMbti);
            log.info("target mbti = {}", targetMbti);
            log.info("text = {}", text);
            String MODEL_SERVER_URL = "http://localhost:8000";
            ResponseEntity<TranslateResponse> response = restTemplate.postForEntity(
                MODEL_SERVER_URL + "/translate",
                request,
                TranslateResponse.class
            );
            log.info("response = {}", response.getBody());

            return response.getBody();
        } catch (Exception e) {
            log.error("번역 요청 실패", e);
            throw new RuntimeException("번역 처리 중 오류 발생", e);
        }
    }
}
