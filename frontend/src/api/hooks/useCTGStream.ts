import { useEffect, useState, useCallback, useRef } from 'react';
import { ctgStreamClient, subscribeToCTGStream } from '../websocket';
import { CTGStreamMessage } from '../types';
import { useAnalysisStore } from '../../store/analysisStore';

export const useCTGStream = (ktgId: string, keepDataOnDisconnect: boolean = false) => {
  const [data, setData] = useState<any[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { addWebSocketData, updatePartialData, flushWebSocketData } = useAnalysisStore();

  // Батчинг: накапливаем данные и отправляем пачками
  const batchRef = useRef<{ ts: number; fhr: number; toco: number | null }[]>([]);
  const batchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const processBatch = useCallback(() => {
    if (batchRef.current.length === 0) return;

    // Отправляем все накопленные данные одной пачкой
    const batch = batchRef.current;
    batchRef.current = [];

    // Обновляем локальное состояние
    setData((prev) => {
      prev.push(...batch);
      return prev;
    });

    // Добавляем данные в store пачкой
    for (const point of batch) {
      addWebSocketData(point.ts, point.fhr, point.toco);
    }

    // Принудительно обновляем store после батча
    flushWebSocketData();
  }, [addWebSocketData, flushWebSocketData]);

  const handleCTGData = useCallback(
    (message: CTGStreamMessage) => {
      // Добавляем в батч
      batchRef.current.push(message.data);

      // Если батч полный (10 точек) или прошло время - обрабатываем
      if (batchRef.current.length >= 10) {
        if (batchTimeoutRef.current) {
          clearTimeout(batchTimeoutRef.current);
          batchTimeoutRef.current = null;
        }
        processBatch();
      } else if (batchTimeoutRef.current === null) {
        // Устанавливаем таймаут на обработку батча
        batchTimeoutRef.current = setTimeout(processBatch, 50); // 50мс максимум
      }

      setIsStreaming(true);
    },
    [processBatch]
  );

  useEffect(() => {
    if (!ktgId) {
      // НИКОГДА не очищаем данные - только останавливаем стриминг
      console.log('[USE_CTG_STREAM] ktgId пустой, но НЕ очищаем данные');
      setIsStreaming(false);
      setError(null);
      return;
    }

    // Подписка на CTG поток
    subscribeToCTGStream(ktgId, handleCTGData);

    // НИКОГДА не очищаем данные при смене КТГ
    console.log('[USE_CTG_STREAM] НЕ очищаем данные при смене КТГ');

    return () => {
      // Очищаем таймаут батча
      if (batchTimeoutRef.current) {
        clearTimeout(batchTimeoutRef.current);
        batchTimeoutRef.current = null;
      }

      // Обрабатываем оставшиеся данные в батче
      if (batchRef.current.length > 0) {
        processBatch();
      }

      // Отписка от событий
      ctgStreamClient.offMessage('ctg_stream', handleCTGData);
    };
  }, [ktgId, handleCTGData, keepDataOnDisconnect, updatePartialData]);

  return {
    data,
    isStreaming,
    error,
    clearData: () => setData([]),
  };
};
