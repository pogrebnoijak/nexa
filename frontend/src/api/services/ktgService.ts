import { httpClient, buildUrl } from '../httpClient';
import { API_ENDPOINTS } from '../constants';
import { ApiResponse, KTGRecord, KTGData, Annotation } from '../types';

export class KTGApiService {
  // Получить список КТГ записей
  async getKTGList(): Promise<ApiResponse<KTGRecord[]>> {
    return httpClient.get<KTGRecord[]>(API_ENDPOINTS.KTG.LIST);
  }

  // Получить детали КТГ записи
  async getKTGDetail(id: string): Promise<ApiResponse<KTGRecord>> {
    const url = buildUrl(API_ENDPOINTS.KTG.DETAIL, { id });
    return httpClient.get<KTGRecord>(url);
  }

  // Создать новую КТГ запись
  async createKTG(data: Partial<KTGRecord>): Promise<ApiResponse<KTGRecord>> {
    return httpClient.post<KTGRecord>(API_ENDPOINTS.KTG.CREATE, data);
  }

  // Обновить КТГ запись
  async updateKTG(id: string, data: Partial<KTGRecord>): Promise<ApiResponse<KTGRecord>> {
    const url = buildUrl(API_ENDPOINTS.KTG.UPDATE, { id });
    return httpClient.put<KTGRecord>(url, data);
  }

  // Удалить КТГ запись
  async deleteKTG(id: string): Promise<ApiResponse<void>> {
    const url = buildUrl(API_ENDPOINTS.KTG.DELETE, { id });
    return httpClient.delete<void>(url);
  }

  // Начать запись КТГ
  async startRecording(id: string): Promise<ApiResponse<void>> {
    const url = buildUrl(API_ENDPOINTS.KTG.START_RECORDING, { id });
    return httpClient.post<void>(url);
  }

  // Остановить запись КТГ
  async stopRecording(id: string): Promise<ApiResponse<void>> {
    const url = buildUrl(API_ENDPOINTS.KTG.STOP_RECORDING, { id });
    return httpClient.post<void>(url);
  }

  // Получить данные КТГ
  async getKTGData(
    id: string,
    startTime?: number,
    endTime?: number
  ): Promise<ApiResponse<KTGData[]>> {
    const url = buildUrl(API_ENDPOINTS.KTG.GET_DATA, { id });
    const params = new URLSearchParams();

    if (startTime) params.append('startTime', startTime.toString());
    if (endTime) params.append('endTime', endTime.toString());

    const fullUrl = params.toString() ? `${url}?${params.toString()}` : url;
    return httpClient.get<KTGData[]>(fullUrl);
  }

  // Добавить аннотацию
  async addAnnotation(
    ktgId: string,
    annotation: Omit<Annotation, 'id'>
  ): Promise<ApiResponse<Annotation>> {
    const url = buildUrl(API_ENDPOINTS.KTG.DETAIL, { id: ktgId });
    return httpClient.post<Annotation>(`${url}/annotations`, annotation);
  }

  // Обновить аннотацию
  async updateAnnotation(
    ktgId: string,
    annotationId: string,
    annotation: Partial<Annotation>
  ): Promise<ApiResponse<Annotation>> {
    const url = buildUrl(API_ENDPOINTS.KTG.DETAIL, { id: ktgId });
    return httpClient.put<Annotation>(`${url}/annotations/${annotationId}`, annotation);
  }

  // Удалить аннотацию
  async deleteAnnotation(ktgId: string, annotationId: string): Promise<ApiResponse<void>> {
    const url = buildUrl(API_ENDPOINTS.KTG.DETAIL, { id: ktgId });
    return httpClient.delete<void>(`${url}/annotations/${annotationId}`);
  }
}

// Создаем экземпляр сервиса
export const ktgService = new KTGApiService();
