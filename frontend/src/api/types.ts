// Базовые типы для API
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

// КТГ типы
export interface KTGRecord {
  id: string;
  patientId: string;
  patientName: string;
  weeks: string;
  status: 'recording' | 'completed' | 'paused';
  startTime: string;
  endTime?: string;
  duration?: number; // в минутах
  fetalHR?: number;
  maternalHR?: number;
  variability?: number;
  accelerations?: number;
  decelerations?: number;
}

export interface KTGData {
  timestamp: number;
  fetalHR: number;
  maternalHR: number;
  uterineTone: number;
  annotations?: Annotation[];
  ktgId?: string; // Добавляем ktgId для WebSocket сообщений
}

export interface Annotation {
  id: string;
  time: string;
  event: string;
  value?: string;
  type: 'patient' | 'system' | 'doctor';
}

// Пациент типы
export interface Patient {
  id: string;
  name: string;
  age: number;
  weeks: number;
  medicalHistory?: string;
  riskFactors?: string[];
  lastVisit?: string;
}

// Анализ типы
export interface RiskAnalysis {
  title: string;
  percentage: number;
  color: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
}

export interface Forecast {
  title: string;
  description: string;
  confidence: number;
  timeframe: string;
}

export interface Indicators {
  fetalHR: number;
  maternalHR: number;
  basalHR: number;
  variability: number;
  accelerations: number;
  decelerations: {
    early: number;
    late: number;
    variable: number;
  };
}

// WebSocket типы
export interface WSMessage {
  type: string;
  data: any;
  timestamp: number;
}

export interface KTGDataMessage extends WSMessage {
  type: 'ktg_data';
  data: KTGData;
}

export interface KTGStatusMessage extends WSMessage {
  type: 'ktg_status';
  data: {
    ktgId: string;
    status: KTGRecord['status'];
  };
}

// CTG поток данных (формат от бэкенда)
export interface CTGStreamData {
  ts: number; // timestamp
  fhr: number; // fetal heart rate
  toco: number | null; // uterine tone (toco)
}

// CTG поток данных
export interface CTGStreamMessage extends WSMessage {
  type: 'ctg_stream';
  data: CTGStreamData;
}

// HTTP клиент типы
export interface RequestConfig {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  url: string;
  data?: any;
  headers?: Record<string, string>;
  timeout?: number;
  signal?: AbortSignal;
}

export interface HttpClient {
  get<T>(url: string, config?: Partial<RequestConfig>): Promise<ApiResponse<T>>;
  post<T>(url: string, data?: any, config?: Partial<RequestConfig>): Promise<ApiResponse<T>>;
  put<T>(url: string, data?: any, config?: Partial<RequestConfig>): Promise<ApiResponse<T>>;
  delete<T>(url: string, config?: Partial<RequestConfig>): Promise<ApiResponse<T>>;
  patch<T>(url: string, data?: any, config?: Partial<RequestConfig>): Promise<ApiResponse<T>>;
}
