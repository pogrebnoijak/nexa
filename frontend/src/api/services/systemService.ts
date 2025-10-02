import { httpClient } from '../httpClient';
import { API_ENDPOINTS } from '../constants';
import { ApiResponse } from '../types';

export class SystemApiService {
  // Ping запрос для проверки доступности сервера
  async ping(): Promise<ApiResponse<any>> {
    return httpClient.get<any>(API_ENDPOINTS.SYSTEM.PING);
  }
}

// Создаем экземпляр сервиса
export const systemService = new SystemApiService();

