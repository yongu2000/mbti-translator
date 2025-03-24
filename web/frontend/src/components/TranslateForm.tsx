"use client";

import { useState } from 'react';
import { translateText } from '../api/translate';
import { TranslateRequest } from '../types/translate';

const MBTI_TYPES = [
    'ISTJ', 'ISFJ', 'INFJ', 'INTJ',
    'ISTP', 'ISFP', 'INFP', 'INTP',
    'ESTP', 'ESFP', 'ENFP', 'ENTP',
    'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
];

export default function TranslateForm() {
    const [sourceMbti, setSourceMbti] = useState('');
    const [targetMbti, setTargetMbti] = useState('');
    const [text, setText] = useState('');
    const [translatedText, setTranslatedText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');

        try {
            const request: TranslateRequest = {
                sourceMbti,
                targetMbti,
                text
            };

            const response = await translateText(request);
            setTranslatedText(response.translatedText);
        } catch (err) {
            setError('번역 중 오류가 발생했습니다.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto p-4">
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700">
                        원본 MBTI
                    </label>
                    <select
                        value={sourceMbti}
                        onChange={(e) => setSourceMbti(e.target.value)}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        required
                    >
                        <option value="">선택하세요</option>
                        {MBTI_TYPES.map((type) => (
                            <option key={type} value={type}>
                                {type}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700">
                        목표 MBTI
                    </label>
                    <select
                        value={targetMbti}
                        onChange={(e) => setTargetMbti(e.target.value)}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        required
                    >
                        <option value="">선택하세요</option>
                        {MBTI_TYPES.map((type) => (
                            <option key={type} value={type}>
                                {type}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700">
                        변환할 텍스트
                    </label>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                        rows={4}
                        required
                    />
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                >
                    {isLoading ? '변환 중...' : '변환하기'}
                </button>
            </form>

            {error && (
                <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-md">
                    {error}
                </div>
            )}

            {translatedText && (
                <div className="mt-4">
                    <h3 className="text-lg font-medium text-gray-900">변환 결과</h3>
                    <div className="mt-2 p-4 bg-gray-50 rounded-md">
                        {translatedText}
                    </div>
                </div>
            )}
        </div>
    );
} 