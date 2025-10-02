import { useState, useEffect, useCallback } from 'react';
import { ktgService } from '../services/ktgService';
import { KTGRecord, KTGData, Annotation, ApiResponse } from '../types';

export const useKTGList = () => {
  const [data, setData] = useState<KTGRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await ktgService.getKTGList();
      if (response.success && response.data) {
        setData(response.data);
      } else {
        setError(response.error || 'Failed to fetch KTG list');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useKTGDetail = (id: string) => {
  const [data, setData] = useState<KTGRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (!id) return;

    setLoading(true);
    setError(null);

    try {
      const response = await ktgService.getKTGDetail(id);
      if (response.success && response.data) {
        setData(response.data);
      } else {
        setError(response.error || 'Failed to fetch KTG detail');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useKTGData = (id: string, startTime?: number, endTime?: number) => {
  const [data, setData] = useState<KTGData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (!id) return;

    setLoading(true);
    setError(null);

    try {
      const response = await ktgService.getKTGData(id, startTime, endTime);
      if (response.success && response.data) {
        setData(response.data);
      } else {
        setError(response.error || 'Failed to fetch KTG data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [id, startTime, endTime]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useKTGActions = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startRecording = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ktgService.startRecording(id);
      if (!response.success) {
        setError(response.error || 'Failed to start recording');
      }
      return response.success;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const stopRecording = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ktgService.stopRecording(id);
      if (!response.success) {
        setError(response.error || 'Failed to stop recording');
      }
      return response.success;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const createKTG = useCallback(async (data: Partial<KTGRecord>) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ktgService.createKTG(data);
      if (response.success) {
        return response.data;
      } else {
        setError(response.error || 'Failed to create KTG');
        return null;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    startRecording,
    stopRecording,
    createKTG,
  };
};
