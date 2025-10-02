// Типы для стейт менеджера анализа КТГ

export interface Contraction {
  start: number; // время начала (индекс)
  end: number; // время окончания (индекс)
}

export interface Event {
  kind: 'accel' | 'decel' | 'tachy' | 'brady';
  t_start_s: number; // время начала в секундах
  duration_s: number; // длительность в секундах
  toco_rel?: 'artifact' | 'variable' | 'early' | 'late'; // связь с тонусом
}

export interface Stirring {
  timestamp: number;
  additionalProp1: {
    type: string;
    ts: number;
  };
}

export interface ProbaShort {
  [time: string]: number; // время предсказания -> значение
}

export interface ProbaLong {
  [time: string]: number; // время предсказания -> вероятность гипоксии
}

export interface AnalysisData {
  // Основные показатели
  baseline: number; // базальная ЧСС
  stv: number; // вариабельность (1 мин)
  ltv: number; // вариабельность (10 мин)
  rmssd: number; // вариабельность (1 мин)
  amp: number; // амплитуда осцилляций (30 мин)
  freq: number; // частота осцилляций (30 мин)

  // События
  num_decel: number; // число децелераций
  num_accel: number; // число акцелераций
  contractions: Contraction[]; // схватки
  events: Event[]; // события (акцелерации, децелерации)

  // Качество и прогнозы
  fisher_points: number | null; // фишер
  data_quality: number; // качество данных
  proba_short: ProbaShort | null; // краткосрочный прогноз
  proba_short_cnn: ProbaShort | null; // краткосрочный прогноз (CNN)
  proba_long: ProbaLong | null; // долгосрочный прогноз

  // Дополнительные данные
  ts: number[] | null; // временные метки
  fhr: number[] | null; // ЧСС плода
  toco: (number | null)[] | null; // тонус матки
  stirrings: Stirring[]; // шевеления
  meta: any | null; // метаданные
}

export interface AnalysisState {
  data: AnalysisData | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: number | null;
}

export interface AnalysisActions {
  setAnalysisData: (data: AnalysisData) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearData: () => void;
  updatePartialData: (partialData: Partial<AnalysisData>) => void;
  addWebSocketData: (ts: number, fhr: number, toco: number | null) => void;
  flushWebSocketData: () => void;
}
