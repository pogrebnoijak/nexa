import { API_CONFIG, WS_EVENTS, WS_ENDPOINTS } from './constants';
import {
  WSMessage,
  KTGDataMessage,
  KTGStatusMessage,
  CTGStreamMessage,
  CTGStreamData,
} from './types';

export type WSMessageHandler<T = any> = (message: T) => void;
export type WSConnectionHandler = () => void;
export type WSErrorHandler = (error: Event) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = API_CONFIG.RETRY_ATTEMPTS;
  private reconnectInterval: number = 3000; // 3 секунды
  private isConnecting: boolean = false;

  // Обработчики событий
  private messageHandlers: Map<string, WSMessageHandler[]> = new Map();
  private connectionHandlers: WSConnectionHandler[] = [];
  private disconnectionHandlers: WSConnectionHandler[] = [];
  private errorHandlers: WSErrorHandler[] = [];

  constructor(url: string = API_CONFIG.WS_URL) {
    this.url = url;
  }

  // Подключение к WebSocket
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        reject(new Error('Connection already in progress'));
        return;
      }

      this.isConnecting = true;

      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.connectionHandlers.forEach((handler) => handler());
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            // Заменяем NaN на null перед парсингом (NaN недопустим в JSON)
            const sanitizedData = event.data.replace(/:\s*NaN\b/g, ':null');

            // Парсим данные как CTGStreamData
            const ctgData: CTGStreamData = JSON.parse(sanitizedData);

            // Создаем сообщение в нашем формате
            const message: CTGStreamMessage = {
              type: 'ctg_stream',
              data: ctgData,
              timestamp: Date.now(),
            };

            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          this.isConnecting = false;
          this.disconnectionHandlers.forEach((handler) => handler());

          // Автоматическое переподключение
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.errorHandlers.forEach((handler) => handler(error));
          reject(error);
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  // Отключение от WebSocket
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  // Отправка сообщения
  send(message: WSMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  // Подписка на сообщения определенного типа
  onMessage<T = any>(messageType: string, handler: WSMessageHandler<T>): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType)!.push(handler);
  }

  // Отписка от сообщений
  offMessage(messageType: string, handler: WSMessageHandler): void {
    const handlers = this.messageHandlers.get(messageType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  // Подписка на подключение
  onConnect(handler: WSConnectionHandler): void {
    this.connectionHandlers.push(handler);
  }

  // Подписка на отключение
  onDisconnect(handler: WSConnectionHandler): void {
    this.disconnectionHandlers.push(handler);
  }

  // Подписка на ошибки
  onError(handler: WSErrorHandler): void {
    this.errorHandlers.push(handler);
  }

  // Получение состояния подключения
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // Приватные методы
  private handleMessage(message: WSMessage): void {
    const handlers = this.messageHandlers.get(message.type);
    if (handlers) {
      handlers.forEach((handler) => handler(message));
    }
  }

  private scheduleReconnect(): void {
    setTimeout(() => {
      this.reconnectAttempts++;
      this.connect().catch((error) => {
        console.error('Reconnection failed:', error);
      });
    }, this.reconnectInterval);
  }
}

// Создаем экземпляр WebSocket клиента
export const wsClient = new WebSocketClient();

// Создаем отдельный клиент для CTG потока
export const ctgStreamClient = new WebSocketClient(
  `${API_CONFIG.WS_URL}${WS_ENDPOINTS.CTG_STREAM}`
);

// Утилиты для работы с WebSocket
export const subscribeToKTGData = (
  ktgId: string,
  handler: WSMessageHandler<KTGDataMessage>
): void => {
  wsClient.onMessage('ktg_data', (message: KTGDataMessage) => {
    // Проверяем, что сообщение относится к нужному КТГ
    if (message.data && message.data.ktgId === ktgId) {
      handler(message);
    }
  });
};

export const subscribeToKTGStatus = (
  ktgId: string,
  handler: WSMessageHandler<KTGStatusMessage>
): void => {
  wsClient.onMessage('ktg_status', (message: KTGStatusMessage) => {
    if (message.data && message.data.ktgId === ktgId) {
      handler(message);
    }
  });
};

export const subscribeToCTGStream = (
  ktgId: string,
  handler: WSMessageHandler<CTGStreamMessage>
): void => {
  ctgStreamClient.onMessage('ctg_stream', (message: CTGStreamMessage) => {
    // Поскольку формат бэкенда не содержит ktgId, обрабатываем все сообщения
    handler(message);
  });
};
