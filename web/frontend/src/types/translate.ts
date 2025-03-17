export interface TranslateRequest {
    sourceMbti: string;
    targetMbti: string;
    text: string;
}

export interface TranslateResponse {
    translatedText: string;
} 