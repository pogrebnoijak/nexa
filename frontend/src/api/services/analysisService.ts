import { httpClient, buildUrl } from '../httpClient';
import { API_ENDPOINTS } from '../constants';
import { ApiResponse, RiskAnalysis, Forecast, Indicators } from '../types';

export class AnalysisApiService {
  // Получить список доступных КТГ ID
  async getAvailableIds(): Promise<ApiResponse<string[]>> {
    const url = '/analyze/ids';
    return httpClient.get<string[]>(url);
  }

  // Получить анализ по конкретному ID
  async getAnalysisById(ktgId: string, signal?: AbortSignal): Promise<ApiResponse<any>> {
    const url = `/analyze/ids/${ktgId}?models=`;
    return httpClient.get<any>(url, { signal });
  }

  // Сохранить данные КТГ (только для создания новых КТГ)
  async saveKTGData(
    ktgId: string,
    data: {
      ts: number[];
      fhr: (number | null)[];
      toco: (number | null)[];
      stirrings: any[];
      meta: any;
    }
  ): Promise<ApiResponse<any>> {
    // ПРОВЕРКА: Если передан ID (не пустой), НЕ делаем запрос
    if (ktgId && ktgId.trim() !== '') {
      return {
        success: false,
        error: 'POST запросы к /analyze/ids/{id} запрещены',
        data: null,
      };
    }

    // Всегда используем /analyze/ids (только для создания новых КТГ)
    const url = '/analyze/ids';
    return httpClient.post<any>(url, data);
  }

  // Получить полный анализ
  async getAnalysis(ktgId: string, signal?: AbortSignal): Promise<ApiResponse<any>> {
    const url = `${API_ENDPOINTS.ANALYSIS.ANALYZE}?records=1&predicts=1`;
    return httpClient.get<any>(url, { signal });
  }

  // Получить быстрый анализ без моделей (основные данные + records, но без тяжелых моделей)
  async getAnalysisFast(ktgId: string, signal?: AbortSignal): Promise<ApiResponse<any>> {
    const url = `${API_ENDPOINTS.ANALYSIS.ANALYZE}?records=1&models=null`;
    return httpClient.get<any>(url, { signal });
  }

  // Получить только predicts (прогнозы и графики) через отдельный endpoint
  async getAnalysisPredicts(ktgId: string, signal?: AbortSignal): Promise<ApiResponse<any>> {
    const url = `/analyze/predicts/${ktgId}`;
    return httpClient.get<any>(url, { signal });
  }

  // Получить анализ рисков
  async getRisks(ktgId: string): Promise<ApiResponse<RiskAnalysis[]>> {
    const url = buildUrl(API_ENDPOINTS.ANALYSIS.RISKS, { ktgId });
    return httpClient.get<RiskAnalysis[]>(url);
  }

  // Получить прогнозы
  async getForecasts(ktgId: string): Promise<ApiResponse<Forecast[]>> {
    const url = buildUrl(API_ENDPOINTS.ANALYSIS.FORECAST, { ktgId });
    return httpClient.get<Forecast[]>(url);
  }

  // Получить показатели
  async getIndicators(ktgId: string): Promise<ApiResponse<Indicators>> {
    const url = buildUrl(API_ENDPOINTS.ANALYSIS.INDICATORS, { ktgId });
    return httpClient.get<Indicators>(url);
  }

  // Обновить анализ рисков
  async updateRisks(ktgId: string, risks: RiskAnalysis[]): Promise<ApiResponse<RiskAnalysis[]>> {
    const url = buildUrl(API_ENDPOINTS.ANALYSIS.RISKS, { ktgId });
    return httpClient.put<RiskAnalysis[]>(url, risks);
  }

  // Обновить прогнозы
  async updateForecasts(ktgId: string, forecasts: Forecast[]): Promise<ApiResponse<Forecast[]>> {
    const url = buildUrl(API_ENDPOINTS.ANALYSIS.FORECAST, { ktgId });
    return httpClient.put<Forecast[]>(url, forecasts);
  }
}

// Создаем экземпляр сервиса
export const analysisService = new AnalysisApiService();
