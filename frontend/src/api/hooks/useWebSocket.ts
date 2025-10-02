import { useEffect, useCallback, useState } from 'react';
import { wsClient, subscribeToKTGData, subscribeToKTGStatus } from '../websocket';
import { KTGDataMessage, KTGStatusMessage } from '../types';

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Подключение к WebSocket
    wsClient
      .connect()
      .then(() => {
        setIsConnected(true);
        setError(null);
      })
      .catch((err) => {
        setError(err.message);
        setIsConnected(false);
      });

    // Обработчики событий
    const handleConnect = () => {
      setIsConnected(true);
      setError(null);
    };

    const handleDisconnect = () => {
      setIsConnected(false);
    };

    const handleError = (err: Event) => {
      setError('WebSocket connection error');
      setIsConnected(false);
    };

    wsClient.onConnect(handleConnect);
    wsClient.onDisconnect(handleDisconnect);
    wsClient.onError(handleError);

    // Очистка при размонтировании
    return () => {
      wsClient.disconnect();
    };
  }, []);

  return { isConnected, error };
};

export const useKTGDataStream = (ktgId: string) => {
  const [data, setData] = useState<any[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (!ktgId) return;

    const handleKTGData = (message: KTGDataMessage) => {
      setData((prev) => [...prev, message.data]);
      setIsStreaming(true);
    };

    const handleKTGStatus = (message: KTGStatusMessage) => {
      if (message.data.status === 'completed' || message.data.status === 'paused') {
        setIsStreaming(false);
      }
    };

    // Подписка на данные КТГ
    subscribeToKTGData(ktgId, handleKTGData);
    subscribeToKTGStatus(ktgId, handleKTGStatus);

    // Очистка данных при смене КТГ
    setData([]);
    setIsStreaming(false);

    return () => {
      // Отписка от событий
      wsClient.offMessage('ktg_data', handleKTGData);
      wsClient.offMessage('ktg_status', handleKTGStatus);
    };
  }, [ktgId]);

  return { data, isStreaming };
};

export const useKTGStatus = (ktgId: string) => {
  const [status, setStatus] = useState<'recording' | 'completed' | 'paused'>('paused');

  useEffect(() => {
    if (!ktgId) return;

    const handleStatusUpdate = (message: KTGStatusMessage) => {
      setStatus(message.data.status);
    };

    subscribeToKTGStatus(ktgId, handleStatusUpdate);

    return () => {
      wsClient.offMessage('ktg_status', handleStatusUpdate);
    };
  }, [ktgId]);

  return status;
};
