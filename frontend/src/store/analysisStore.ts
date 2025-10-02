import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { AnalysisState, AnalysisActions, AnalysisData } from './types';

interface AnalysisStore extends AnalysisState, AnalysisActions {}

export const useAnalysisStore = create<AnalysisStore>()(
  devtools(
    (set, get) => ({
      // Начальное состояние
      data: null,
      isLoading: false,
      error: null,
      lastUpdated: null,

      // Действия
      setAnalysisData: (data: AnalysisData) => {
        set({
          data,
          error: null,
          lastUpdated: Date.now(),
        });
      },

      setLoading: (loading: boolean) => {
        set({ isLoading: loading });
      },

      setError: (error: string | null) => {
        set({
          error,
          isLoading: false,
        });
      },

      clearData: () => {
        set({
          data: null,
          error: null,
          lastUpdated: null,
        });
      },

      updatePartialData: (partialData: Partial<AnalysisData>) => {
        const currentData = get().data;
        if (currentData) {
          set({
            data: { ...currentData, ...partialData },
            lastUpdated: Date.now(),
          });
        }
      },

      addWebSocketData: (ts: number, fhr: number, toco: number | null) => {
        const currentData = get().data;

        // Если данных нет, создаем новые массивы
        if (!currentData) {
          set({
            data: {
              baseline: 0,
              stv: 0,
              ltv: 0,
              rmssd: 0,
              amp: 0,
              freq: 0,
              num_decel: 0,
              num_accel: 0,
              contractions: [],
              events: [],
              fisher_points: null,
              data_quality: 1.0,
              proba_short: null,
              proba_short_cnn: null,
              proba_long: null,
              stirrings: [],
              meta: null,
              ts: [ts],
              fhr: [fhr],
              toco: [toco],
            } as AnalysisData,
            lastUpdated: Date.now(),
          });
          return;
        }

        // Простое добавление данных (батчинг происходит на уровне WebSocket)
        const existingTs = currentData.ts || [];
        const existingFhr = currentData.fhr || [];
        const existingToco = currentData.toco || [];

        // Добавляем данные в массивы
        existingTs.push(ts);
        existingFhr.push(fhr);
        existingToco.push(toco);

        // Обновляем store сразу (батчинг уже произошел на уровне WebSocket)
        set({
          data: {
            ...currentData,
            ts: existingTs,
            fhr: existingFhr,
            toco: existingToco,
          },
          lastUpdated: Date.now(),
        });
      },

      // Принудительное обновление store (для завершения потока)
      flushWebSocketData: () => {
        const currentData = get().data;
        if (currentData) {
          set({
            data: {
              ...currentData,
              ts: currentData.ts,
              fhr: currentData.fhr,
              toco: currentData.toco,
            },
            lastUpdated: Date.now(),
          });
        }
      },
    }),
    {
      name: 'analysis-store', // имя для devtools
    }
  )
);

// Селекторы для удобного использования
export const useAnalysisData = () => useAnalysisStore((state) => state.data);
export const useAnalysisLoading = () => useAnalysisStore((state) => state.isLoading);
export const useAnalysisError = () => useAnalysisStore((state) => state.error);
export const useAnalysisLastUpdated = () => useAnalysisStore((state) => state.lastUpdated);

// Селекторы для конкретных показателей
export const useBaseline = () => useAnalysisStore((state) => state.data?.baseline);
export const useVariability = () =>
  useAnalysisStore((state) => ({
    stv: state.data?.stv,
    ltv: state.data?.ltv,
    rmssd: state.data?.rmssd,
  }));
export const useOscillations = () =>
  useAnalysisStore((state) => ({
    amp: state.data?.amp,
    freq: state.data?.freq,
  }));
export const useEvents = () =>
  useAnalysisStore((state) => ({
    num_decel: state.data?.num_decel,
    num_accel: state.data?.num_accel,
    events: state.data?.events,
  }));
export const useContractions = () => useAnalysisStore((state) => state.data?.contractions);
export const useQuality = () =>
  useAnalysisStore((state) => ({
    data_quality: state.data?.data_quality,
    fisher_points: state.data?.fisher_points,
  }));
export const useForecasts = () =>
  useAnalysisStore((state) => ({
    proba_short: state.data?.proba_short,
    proba_short_cnn: state.data?.proba_short_cnn,
    proba_long: state.data?.proba_long,
  }));
