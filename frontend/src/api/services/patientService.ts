import { httpClient, buildUrl } from '../httpClient';
import { API_ENDPOINTS } from '../constants';
import { ApiResponse, Patient } from '../types';

export class PatientApiService {
  // Получить список пациентов
  async getPatients(): Promise<ApiResponse<Patient[]>> {
    return httpClient.get<Patient[]>(API_ENDPOINTS.PATIENTS.LIST);
  }

  // Получить детали пациента
  async getPatient(id: string): Promise<ApiResponse<Patient>> {
    const url = buildUrl(API_ENDPOINTS.PATIENTS.DETAIL, { id });
    return httpClient.get<Patient>(url);
  }

  // Создать нового пациента
  async createPatient(data: Omit<Patient, 'id'>): Promise<ApiResponse<Patient>> {
    return httpClient.post<Patient>(API_ENDPOINTS.PATIENTS.CREATE, data);
  }

  // Обновить данные пациента
  async updatePatient(id: string, data: Partial<Patient>): Promise<ApiResponse<Patient>> {
    const url = buildUrl(API_ENDPOINTS.PATIENTS.UPDATE, { id });
    return httpClient.put<Patient>(url, data);
  }

  // Удалить пациента
  async deletePatient(id: string): Promise<ApiResponse<void>> {
    const url = buildUrl(API_ENDPOINTS.PATIENTS.DELETE, { id });
    return httpClient.delete<void>(url);
  }

  // Поиск пациентов
  async searchPatients(query: string): Promise<ApiResponse<Patient[]>> {
    const params = new URLSearchParams({ q: query });
    return httpClient.get<Patient[]>(`${API_ENDPOINTS.PATIENTS.LIST}?${params.toString()}`);
  }
}

// Создаем экземпляр сервиса
export const patientService = new PatientApiService();
