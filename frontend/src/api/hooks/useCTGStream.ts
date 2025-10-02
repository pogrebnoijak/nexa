import { useEffect, useState, useCallback } from 'react';
import { ctgStreamClient, subscribeToCTGStream } from '../websocket';
import { CTGStreamMessage } from '../types';
import { useAnalysisStore } from '../../store/analysisStore';

export const useCTGStream = (ktgId: string, keepDataOnDisconnect: boolean = false) => {
  const [data, setData] = useState<any[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { addWebSocketData, updatePartialData } = useAnalysisStore();

  const handleCTGData = useCallback(
    (message: CTGStreamMessage) => {
      setData((prev) => [...prev, message.data]);

      // Добавляем новую точку данных в store
      addWebSocketData(message.data.ts, message.data.fhr, message.data.toco);

      setIsStreaming(true);
    },
    [addWebSocketData]
  );

  useEffect(() => {
    if (!ktgId) {
      // Если ktgId пустой и не нужно сохранять данные, сбрасываем их
      if (!keepDataOnDisconnect) {
        setData([]);
        setIsStreaming(false);
        setError(null);
        // Очищаем данные графика в store
        updatePartialData({
          ts: null,
          fhr: null,
          toco: null,
        });
      } else {
        // Просто останавливаем стриминг, но данные сохраняем
        setIsStreaming(false);
        setError(null);
      }
      return;
    }

    // Подписка на CTG поток
    subscribeToCTGStream(ktgId, handleCTGData);

    // Очистка данных при смене КТГ (но не при завершении записи)
    if (!keepDataOnDisconnect) {
      setData([]);
      setIsStreaming(false);
      setError(null);
    }

    return () => {
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
