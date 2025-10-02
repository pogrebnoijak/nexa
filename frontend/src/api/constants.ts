// API константы
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000',
  TIMEOUT: 50000, // 10 секунд
  RETRY_ATTEMPTS: 3,
} as const;

// HTTP методы
export const HTTP_METHODS = {
  GET: 'GET',
  POST: 'POST',
  PUT: 'PUT',
  DELETE: 'DELETE',
  PATCH: 'PATCH',
} as const;

// WebSocket события
export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  KTG_DATA: 'ktg_data',
  KTG_STATUS: 'ktg_status',
  PATIENT_UPDATE: 'patient_update',
  CTG_STREAM: 'ctg_stream',
} as const;

// WebSocket эндпоинты
export const WS_ENDPOINTS = {
  CTG_STREAM: '/stream/ctg',
} as const;

// API эндпоинты
export const API_ENDPOINTS = {
  // Системные эндпоинты
  SYSTEM: {
    PING: '/ping',
  },
  // КТГ данные
  KTG: {
    LIST: '/api/ktg',
    DETAIL: '/api/ktg/:id',
    CREATE: '/api/ktg',
    UPDATE: '/api/ktg/:id',
    DELETE: '/api/ktg/:id',
    START_RECORDING: '/api/ktg/:id/start',
    STOP_RECORDING: '/api/ktg/:id/stop',
    GET_DATA: '/api/ktg/:id/data',
  },
  // Пациенты
  PATIENTS: {
    LIST: '/api/patients',
    DETAIL: '/api/patients/:id',
    CREATE: '/api/patients',
    UPDATE: '/api/patients/:id',
    DELETE: '/api/patients/:id',
  },
  // Анализ данных
  ANALYSIS: {
    ANALYZE: '/analyze',
    RISKS: '/api/analysis/risks/:ktgId',
    FORECAST: '/api/analysis/forecast/:ktgId',
    INDICATORS: '/api/analysis/indicators/:ktgId',
  },
} as const;

// Статусы ответов
export const API_STATUS = {
  SUCCESS: 'success',
  ERROR: 'error',
  LOADING: 'loading',
} as const;
