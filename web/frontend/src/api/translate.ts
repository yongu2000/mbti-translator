import { TranslateRequest, TranslateResponse } from '../types/translate';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export const translateText = async (request: TranslateRequest): Promise<TranslateResponse> => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error('번역 요청 실패');
        }

        return await response.json();
    } catch (error) {
        console.error('번역 API 오류:', error);
        throw error;
    }
}; 