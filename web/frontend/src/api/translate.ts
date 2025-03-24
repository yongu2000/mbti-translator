import { TranslateRequest, TranslateResponse } from '../types/translate';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

export const translateText = async (request: TranslateRequest): Promise<TranslateResponse> => {
    try {
        console.log('API 요청 URL:', `${API_BASE_URL}/api/translate`);
        console.log('요청 데이터:', request);
        
        const response = await fetch(`${API_BASE_URL}/api/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        console.log('응답 상태:', response.status);
        
        if (!response.ok) {
            throw new Error('번역 요청 실패');
        }

        const data = await response.json();
        console.log('응답 데이터:', data);
        return data;
    } catch (error) {
        console.error('번역 API 오류:', error);
        throw error;
    }
}; 