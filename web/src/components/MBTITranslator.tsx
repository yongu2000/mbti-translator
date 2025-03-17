'use client';

import { useState } from 'react';

const MBTI_TYPES = [
  'ISTJ', 'ISFJ', 'INFJ', 'INTJ',
  'ISTP', 'ISFP', 'INFP', 'INTP',
  'ESTP', 'ESFP', 'ENFP', 'ENTP',
  'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
];

export default function MBTITranslator() {
  const [myMBTI, setMyMBTI] = useState('');
  const [targetMBTI, setTargetMBTI] = useState('');
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');

  const handleTranslate = async () => {
    // TODO: API 연동 후 실제 번역 로직 구현
    setTranslatedText(`${myMBTI}에서 ${targetMBTI}로 번역된 결과가 여기에 표시됩니다.`);
  };

  return (
    <div className="space-y-6 p-6 bg-white rounded-lg shadow-md">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            내 MBTI
          </label>
          <select
            value={myMBTI}
            onChange={(e) => setMyMBTI(e.target.value)}
            className="w-full p-2 border rounded-md"
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
          <label className="block text-sm font-medium text-gray-700 mb-2">
            변환할 MBTI
          </label>
          <select
            value={targetMBTI}
            onChange={(e) => setTargetMBTI(e.target.value)}
            className="w-full p-2 border rounded-md"
          >
            <option value="">선택하세요</option>
            {MBTI_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          변환할 텍스트
        </label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full p-2 border rounded-md h-32"
          placeholder="변환하고 싶은 텍스트를 입력하세요..."
        />
      </div>

      <button
        onClick={handleTranslate}
        disabled={!myMBTI || !targetMBTI || !inputText}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        변환하기
      </button>

      {translatedText && (
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            변환 결과
          </label>
          <div className="p-4 bg-gray-50 rounded-md min-h-[8rem] whitespace-pre-wrap">
            {translatedText}
          </div>
        </div>
      )}
    </div>
  );
} 