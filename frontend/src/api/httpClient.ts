import { API_CONFIG, HTTP_METHODS } from './constants';
import { ApiResponse, ApiError, RequestConfig, HttpClient } from './types';

class HttpClientImpl implements HttpClient {
  private baseURL: string;
  private defaultTimeout: number;

  constructor(baseURL: string = API_CONFIG.BASE_URL, timeout: number = API_CONFIG.TIMEOUT) {
    this.baseURL = baseURL;
    this.defaultTimeout = timeout;
  }

  private async makeRequest<T>(
    method: string,
    url: string,
    data?: any,
    config?: Partial<RequestConfig>
  ): Promise<ApiResponse<T>> {
    const fullUrl = url.startsWith('http') ? url : `${this.baseURL}${url}`;
    console.log(`[HTTP CLIENT] ${method} запрос к: ${fullUrl}`);
    const timeout = config?.timeout || this.defaultTimeout;

    const requestConfig: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...config?.headers,
      },
      signal: AbortSignal.timeout(timeout),
    };

    if (data && method !== HTTP_METHODS.GET) {
      requestConfig.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(fullUrl, requestConfig);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const responseData = await response.json();

      return {
        success: true,
        data: responseData,
      };
    } catch (error) {
      const apiError: ApiError = {
        message: error instanceof Error ? error.message : 'Unknown error',
        code: 'NETWORK_ERROR',
      };

      return {
        success: false,
        error: apiError.message,
      };
    }
  }

  async get<T>(url: string, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.makeRequest<T>(HTTP_METHODS.GET, url, undefined, config);
  }

  async post<T>(url: string, data?: any, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.makeRequest<T>(HTTP_METHODS.POST, url, data, config);
  }

  async put<T>(url: string, data?: any, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.makeRequest<T>(HTTP_METHODS.PUT, url, data, config);
  }

  async delete<T>(url: string, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.makeRequest<T>(HTTP_METHODS.DELETE, url, undefined, config);
  }

  async patch<T>(
    url: string,
    data?: any,
    config?: Partial<RequestConfig>
  ): Promise<ApiResponse<T>> {
    return this.makeRequest<T>(HTTP_METHODS.PATCH, url, data, config);
  }
}

// Создаем экземпляр HTTP клиента
export const httpClient = new HttpClientImpl();

// Утилиты для работы с URL
export const buildUrl = (endpoint: string, params?: Record<string, string | number>): string => {
  let url = endpoint;

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url = url.replace(`:${key}`, String(value));
    });
  }

  return url;
};

// Утилиты для обработки ошибок
export const handleApiError = (error: any): ApiError => {
  if (error instanceof Error) {
    return {
      message: error.message,
      code: 'CLIENT_ERROR',
    };
  }

  return {
    message: 'Unknown error occurred',
    code: 'UNKNOWN_ERROR',
  };
};
