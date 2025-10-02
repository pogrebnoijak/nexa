// Экспорт констант
export * from './constants';

// Экспорт типов
export * from './types';

// Экспорт HTTP клиента
export * from './httpClient';

// Экспорт WebSocket клиента
export * from './websocket';

// Экспорт сервисов
export { ktgService } from './services/ktgService';
export { analysisService } from './services/analysisService';
export { patientService } from './services/patientService';
export { systemService } from './services/systemService';

// Экспорт хуков
export * from './hooks/useKTG';
export * from './hooks/useWebSocket';
export * from './hooks/useCTGStream';
export * from './hooks/usePing';

// Экспорт утилит
export { buildUrl, handleApiError } from './httpClient';
export {
  subscribeToKTGData,
  subscribeToKTGStatus,
  subscribeToCTGStream,
  ctgStreamClient,
} from './websocket';
