package com.yongu2000.mbtitranslator.dto;

import lombok.Data;

@Data
public class TranslateRequest {
    private String sourceMbti;
    private String targetMbti;
    private String text;
}
