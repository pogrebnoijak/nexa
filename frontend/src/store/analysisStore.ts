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
        console.log('clearData');
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
        const existingTs = currentData?.ts || [];
        const existingFhr = currentData?.fhr || [];
        const existingToco = currentData?.toco || [];

        set({
          data: {
            ...currentData,
            ts: [...existingTs, ts],
            fhr: [...existingFhr, fhr],
            toco: [...existingToco, toco],
          } as AnalysisData,
          lastUpdated: Date.now(),
        });
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
