import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { Header } from '../../components/layout/Header';
import { Chart } from '../../components/charts/Chart';
import { IndicatorCard } from '../../components/ui/IndicatorCard';
import { FisherCard } from '../../components/ui/FisherCard';
import { AnnotationsPanel } from '../../components/ui/AnnotationsPanel';
import { TimeScale } from '../../components/ui/TimeScale';
import { StatusHeader } from '../../components/ui/StatusHeader';
import { RisksPanel } from '../../components/ui/RisksPanel';
import { ForecastPanel } from '../../components/ui/ForecastPanel';
import { UI_TEXT, CHART_DATA } from '../../shared/constants';
import { useCTGStream, ctgStreamClient, analysisService } from '../../api';
import { useAnalysisStore } from '../../store';
import styles from './HomePage.module.scss';

export const HomePage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const ktgId = id || '123455';
  const isNewKTG = id === 'new'; // Новое КТГ если URL содержит /new
  const [actualKtgId, setActualKtgId] = useState<string>(isNewKTG ? '' : ktgId); // Реальный ID для новых КТГ
  const [isManualDisconnect, setIsManualDisconnect] = useState(false); // Флаг ручного отключения

  const [windowRange, setWindowRange] = useState<{ start: number; end: number }>({
    start: 0,
    end: 50,
  });

  // Состояние КТГ: 'idle' | 'recording' | 'completed' | 'loading'
  const [ktgStatus, setKtgStatus] = useState<'idle' | 'recording' | 'completed' | 'loading'>(
    'loading'
  );

  // Состояние для отслеживания загрузки существующих данных
  const [isLoadingExistingData, setIsLoadingExistingData] = useState(false);

  // Состояние подключения WebSocket
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);

  // Состояние для переключения вариабельности
  const [variabilityMetric, setVariabilityMetric] = useState<'stv' | 'ltv' | 'rmssd'>('stv');

  // Состояние для пользовательских аннотаций (меток)
  const [userAnnotations, setUserAnnotations] = useState<
    Array<{
      time: string;
      event: string;
      value: string;
    }>
  >([]);

  // Стейт менеджер для анализа
  const {
    data: analysisData,
    isLoading: analysisLoading,
    error: analysisError,
    setAnalysisData,
    setLoading,
    setError,
    updatePartialData,
    clearData,
  } = useAnalysisStore();

  // Подписка на CTG поток (подключаемся только во время записи)
  const {
    data: ctgStreamData,
    isStreaming,
    error: streamError,
  } = useCTGStream(actualKtgId || ktgId, ktgStatus === 'completed' || ktgStatus === 'recording');

  // Отладочная информация для useCTGStream
  useEffect(() => {
    console.log('[HOMEPAGE] useCTGStream параметры:', {
      ktgId: actualKtgId || ktgId,
      keepDataOnDisconnect: ktgStatus === 'completed' || ktgStatus === 'recording',
      ktgStatus,
    });
  }, [actualKtgId, ktgId, ktgStatus]);

  // Проверяем, является ли КТГ существующей при загрузке компонента
  useEffect(() => {
    const abortController = new AbortController();

    const checkExistingKTG = async () => {
      if (!ktgId) return;

      // Если это новая КТГ (создана через кнопку "Новое КТГ"), пропускаем API запрос
      if (isNewKTG) {
        console.log(`[HOMEPAGE] Новая КТГ с ID: ${ktgId}, устанавливаем статус idle`);
        setKtgStatus('idle');
        setIsLoadingExistingData(false);
        return;
      }

      try {
        setIsLoadingExistingData(true);
        console.log(`[HOMEPAGE] Проверяем существующую КТГ с ID: ${ktgId}`);

        // Двухэтапная загрузка для существующих КТГ
        console.log('[HOMEPAGE] Этап 1: Быстрая загрузка основных данных');
        const result = await analysisService.getAnalysisById(ktgId, abortController.signal);

        // Проверяем, не был ли запрос отменен
        if (abortController.signal.aborted) {
          console.log('[HOMEPAGE] Запрос отменен при размонтировании');
          return;
        }

        if (result.success && result.data) {
          console.log(`[HOMEPAGE] Найдена существующая КТГ, устанавливаем статус completed`);
          // Устанавливаем статус как завершенную
          setKtgStatus('completed');

          // Сохраняем основные данные анализа в store
          setAnalysisData(result.data);

          // Сразу запрашиваем predicts для графиков
          console.log('[HOMEPAGE] Этап 2: Загрузка predicts и графиков');
          try {
            const predictsResult = await analysisService.getAnalysisPredicts(
              ktgId,
              abortController.signal
            );

            if (predictsResult.success && predictsResult.data) {
              updatePartialData(predictsResult.data);
            }
          } catch (predictsError) {
            console.warn('[HOMEPAGE] Ошибка загрузки predicts:', predictsError);
            // Не критично, основные данные уже загружены
          }

          toast.success(`КТГ ID ${ktgId} загружена из архива`);
        } else {
          console.log(`[HOMEPAGE] КТГ не найдена в архиве, устанавливаем статус idle`);
          // КТГ не существует, устанавливаем статус idle для новой записи
          setKtgStatus('idle');
        }
      } catch (error) {
        // Игнорируем ошибки отмены запроса
        if (error instanceof Error && error.name === 'AbortError') {
          console.log('[HOMEPAGE] Запрос отменен');
          return;
        }

        console.error(`[HOMEPAGE] Ошибка при проверке существующей КТГ:`, error);
        // В случае ошибки устанавливаем статус completed (запись была проведена)
        setKtgStatus('completed');
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoadingExistingData(false);
        }
      }
    };

    checkExistingKTG();

    return () => {
      abortController.abort();
    };
  }, [ktgId, isNewKTG]);

  // Функция сохранения данных КТГ (используется при любом завершении записи)
  const saveKTGData = useCallback(
    async (reason: string) => {
      console.log(`[HOMEPAGE] Сохранение данных КТГ по причине: ${reason}`);

      try {
        // Собираем данные для отправки
        const ktgData = {
          ts: analysisData?.ts || [],
          fhr: analysisData?.fhr || [],
          toco: analysisData?.toco || [],
          stirrings: analysisData?.stirrings || [],
          meta: analysisData?.meta || {},
        };

        console.log('[HOMEPAGE] Отправляем данные КТГ:', ktgData);

        // Используем actualKtgId для новых КТГ, иначе ktgId
        const idToUse = actualKtgId || ktgId;

        // Отправляем данные на сервер
        const result = await analysisService.saveKTGData(idToUse, ktgData);

        if (result.success) {
          console.log('[HOMEPAGE] Данные КТГ успешно сохранены');
          toast.success(`КТГ ID ${idToUse} сохранена (${reason})`);
        } else {
          console.error('[HOMEPAGE] Ошибка сохранения КТГ:', result.error);
          toast.error('Ошибка при сохранении КТГ: ' + (result.error || 'Неизвестная ошибка'));
        }
      } catch (error) {
        console.error('[HOMEPAGE] Ошибка при сохранении КТГ:', error);
        toast.error('Ошибка при сохранении КТГ');
      }
    },
    [ktgId, actualKtgId, analysisData]
  );

  // Очистка при размонтировании компонента
  useEffect(() => {
    // Создаем AbortController для отмены запросов
    const abortController = new AbortController();

    return () => {
      console.log('[HOMEPAGE] Размонтирование компонента, выполняем очистку');

      // Отменяем все активные запросы
      abortController.abort();

      // Отключаем WebSocket если подключен
      if (ctgStreamClient.isConnected) {
        ctgStreamClient.disconnect();
      }

      // НИКОГДА не очищаем данные при размонтировании - только при навигации
      console.log('[HOMEPAGE] НЕ очищаем данные при размонтировании компонента');
    };
  }, [ktgStatus]);

  // Обработчик начала записи
  const handleStartRecording = useCallback(async () => {
    // Если это новое КТГ, создаем его на сервере
    if (isNewKTG && !actualKtgId) {
      try {
        console.log('[HOMEPAGE] Создание нового КТГ на сервере');

        // Создаем новое КТГ с пустыми данными
        const ktgData = {
          ts: [],
          fhr: [],
          toco: [],
          stirrings: [],
          meta: {},
        };

        // POST запрос к /analyze/ids для создания нового КТГ
        const result = await analysisService.saveKTGData('', ktgData); // Пустой ID для создания нового

        if (result.success && result.data?.id) {
          const newId = result.data.id;
          setActualKtgId(newId);

          // Обновляем URL с полученным ID
          navigate(`/ktg/${newId}`, { replace: true });

          toast.success(`Создано новое КТГ с ID: ${newId}`);

          // Устанавливаем статус recording только после успешного создания КТГ
          console.log('[HOMEPAGE] Устанавливаем статус recording для нового КТГ');
          setKtgStatus('recording');

          // Дополнительная принудительная установка через небольшую задержку
          setTimeout(() => {
            console.log('[HOMEPAGE] Принудительная установка статуса для нового КТГ');
            setKtgStatus('recording');
          }, 100);
        } else {
          throw new Error('Сервер не вернул ID нового КТГ');
        }
      } catch (error) {
        console.error('[HOMEPAGE] Ошибка создания нового КТГ:', error);
        toast.error('Ошибка создания нового КТГ');
        return;
      }
    } else {
      // Для существующих КТГ устанавливаем статус recording сразу
      console.log('[HOMEPAGE] Устанавливаем статус recording для существующего КТГ');
      setKtgStatus('recording');

      // Дополнительная принудительная установка через небольшую задержку
      setTimeout(() => {
        console.log('[HOMEPAGE] Принудительная установка статуса для существующего КТГ');
        setKtgStatus('recording');
      }, 100);
    }

    // Подключаемся к WebSocket
    if (!ctgStreamClient.isConnected) {
      try {
        console.log('[HOMEPAGE] Подключение к WebSocket...');
        await ctgStreamClient.connect();
        console.log('[HOMEPAGE] WebSocket подключен успешно');
        setIsWebSocketConnected(true);

        // Обработчик отключения WebSocket
        ctgStreamClient.onDisconnect(async () => {
          console.log('[HOMEPAGE] WebSocket отключен');

          // Для любого отключения просто завершаем запись (не сохраняем данные)
          if (!isManualDisconnect) {
            console.log('[HOMEPAGE] Автоматическое отключение');
            toast.error('Соединение потеряно. Запись завершена.');
          } else {
            console.log('[HOMEPAGE] Ручное отключение');
          }

          // Автоматически завершаем запись при потере соединения
          setKtgStatus('completed');
          setIsWebSocketConnected(false);
          setUserAnnotations([]);
          setIsManualDisconnect(false); // Сбрасываем флаг
        });

        // Обработчик ошибок WebSocket
        ctgStreamClient.onError(async (error) => {
          console.log('[HOMEPAGE] Ошибка WebSocket');

          // Для любой ошибки просто завершаем запись (не сохраняем данные)
          if (!isManualDisconnect) {
            console.log('[HOMEPAGE] Ошибка соединения');
            toast.error('Ошибка соединения. Запись завершена.');
          }

          // Автоматически завершаем запись при ошибке соединения
          setKtgStatus('completed');
          setIsWebSocketConnected(false);
          setUserAnnotations([]);
          setIsManualDisconnect(false); // Сбрасываем флаг
        });

        // Принудительно обновляем состояние через небольшую задержку
        setTimeout(() => {
          setIsWebSocketConnected(ctgStreamClient.isConnected);
        }, 100);
      } catch (err) {
        console.error('[HOMEPAGE] Ошибка подключения к WebSocket:', err);
        setIsWebSocketConnected(false);
        // Не сбрасываем статус, если WebSocket не подключился - запись может продолжаться
        console.log('[HOMEPAGE] WebSocket не подключился, но запись продолжается');
      }
    } else {
      console.log('[HOMEPAGE] WebSocket уже подключен');
      setIsWebSocketConnected(true);
    }

    // Дополнительная проверка статуса через небольшую задержку
    setTimeout(() => {
      console.log('[HOMEPAGE] Финальная проверка - принудительно устанавливаем статус recording');
      setKtgStatus('recording');
    }, 200);
  }, [isNewKTG, actualKtgId, navigate]);

  // Обработчик завершения записи
  const handleCompleteRecording = useCallback(async () => {
    console.log('[HOMEPAGE] Завершение записи КТГ');

    // Устанавливаем флаг ручного отключения
    setIsManualDisconnect(true);

    // Сначала отключаемся от WebSocket
    ctgStreamClient.disconnect();
    setIsWebSocketConnected(false);

    // Не сохраняем данные при завершении записи (данные сохраняются только при создании новых КТГ)
    console.log('[HOMEPAGE] Завершение записи без сохранения данных');

    // Устанавливаем статус завершенной записи
    setKtgStatus('completed');

    // Очищаем пользовательские аннотации
    setUserAnnotations([]);
  }, [isNewKTG]);

  // Обработчик выхода из КТГ (сброс состояния)
  const handleExit = useCallback(() => {
    console.log('[HOMEPAGE] Выход из КТГ, сбрасываем состояние');

    // Отключаем WebSocket если подключен
    if (isWebSocketConnected) {
      ctgStreamClient.disconnect();
      setIsWebSocketConnected(false);
    }

    // Сбрасываем все состояния
    setKtgStatus('idle');
    setUserAnnotations([]);
    setIsLoadingExistingData(false);
    setActualKtgId(isNewKTG ? '' : ktgId); // Сбрасываем actualKtgId
    setIsManualDisconnect(false); // Сбрасываем флаг ручного отключения

    // Очищаем данные анализа в store
    clearData();

    console.log('[HOMEPAGE] Состояние сброшено');
  }, [isWebSocketConnected, clearData, isNewKTG, ktgId]);

  // Обработчик загрузки CSV файла
  const handleUploadCSV = useCallback(
    async (file: File) => {
      console.log('[HOMEPAGE] Загрузка CSV файла:', file.name);

      try {
        const text = await file.text();
        const lines = text.split('\n');

        // Определяем разделитель (запятая или точка с запятой)
        const delimiter = lines[0].includes(';') ? ';' : ',';

        // Проверяем заголовки (первая строка)
        const headers = lines[0].split(delimiter).map((h) => h.trim());
        if (!headers.includes('time_sec') || !headers.includes('value')) {
          toast.error(
            'CSV файл должен содержать колонки time_sec и value (разделитель: запятая или точка с запятой)'
          );
          return;
        }

        // Парсим данные
        const timeIndex = headers.indexOf('time_sec');
        const valueIndex = headers.indexOf('value');
        const tocoIndex = headers.indexOf('toco'); // Индекс колонки toco (может не быть)

        const ts: number[] = [];
        const fhr: number[] = [];
        const tocoRaw: (number | null)[] = [];

        for (let i = 1; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line) continue;

          const values = line.split(delimiter);
          const timeValue = parseFloat(values[timeIndex]);
          const dataValue = parseFloat(values[valueIndex]);
          const tocoValue = tocoIndex >= 0 ? parseFloat(values[tocoIndex]) : null;

          if (!isNaN(timeValue) && !isNaN(dataValue)) {
            ts.push(timeValue);
            fhr.push(dataValue);
            tocoRaw.push(!isNaN(tocoValue!) ? tocoValue : null);
          }
        }

        // Обрабатываем данные toco: если есть валидные значения, используем их, иначе null
        const hasValidToco = tocoRaw.some((val) => val !== null);
        const toco = hasValidToco ? (tocoRaw.filter((val) => val !== null) as number[]) : null;

        if (ts.length === 0) {
          toast.error('Не удалось найти валидные данные в CSV файле');
          return;
        }

        // Загружаем данные в store с полной структурой AnalysisData
        const csvData = {
          // Основные показатели (заглушки для CSV данных)
          baseline: 0,
          stv: 0,
          ltv: 0,
          rmssd: 0,
          amp: 0,
          freq: 0,

          // События (пустые для CSV данных)
          num_decel: 0,
          num_accel: 0,
          contractions: [],
          events: [],

          // Качество и прогнозы (заглушки)
          fisher_points: null,
          data_quality: 1.0,
          proba_short: null,
          proba_short_cnn: null,
          proba_long: null,

          // Основные данные из CSV
          ts,
          fhr,
          toco,
          stirrings: [],
          meta: { source: 'csv_upload', filename: file.name },
        };

        setAnalysisData(csvData);
        setKtgStatus('completed');

        // Для новых КТГ создаем КТГ на сервере и получаем ID
        if (isNewKTG && !actualKtgId) {
          try {
            console.log('[HOMEPAGE] Создание нового КТГ для загруженного CSV');

            // Создаем новое КТГ с загруженными данными
            const ktgData = {
              ts,
              fhr,
              toco: toco || [],
              stirrings: [],
              meta: {},
            };

            // POST запрос к /analyze/ids для создания нового КТГ
            const result = await analysisService.saveKTGData('', ktgData);

            if (result.success && result.data?.id) {
              const newId = result.data.id;
              setActualKtgId(newId);

              // Обновляем URL с полученным ID
              navigate(`/ktg/${newId}`, { replace: true });

              const tocoMessage = tocoIndex >= 0 ? ' (включая данные TOCO)' : '';
              toast.success(
                `Создано новое КТГ с ID: ${newId}. Загружено ${ts.length} точек данных из ${file.name}${tocoMessage}`
              );
            } else {
              throw new Error('Сервер не вернул ID нового КТГ');
            }
          } catch (error) {
            console.error('[HOMEPAGE] Ошибка создания КТГ для CSV:', error);
            toast.error('Ошибка создания КТГ для загруженных данных');
          }
        } else {
          // Для существующих КТГ просто загружаем данные (не сохраняем на сервер)
          console.log('[HOMEPAGE] Загрузка CSV для существующего КТГ (без сохранения на сервер)');

          const tocoMessage = tocoIndex >= 0 ? ' (включая данные TOCO)' : '';
          toast.success(`Загружено ${ts.length} точек данных из ${file.name}${tocoMessage}`);
        }
      } catch (error) {
        console.error('[HOMEPAGE] Ошибка при загрузке CSV:', error);
        toast.error('Ошибка при обработке CSV файла');
      }
    },
    [setAnalysisData, isNewKTG, actualKtgId, navigate]
  );

  // Функция добавления пользовательской аннотации
  const addUserAnnotation = useCallback((mark: string, timestamp: number) => {
    // Форматируем время в минуты и секунды
    const timeInMinutes = timestamp / 60;
    const mins = Math.floor(timeInMinutes);
    const secs = Math.round((timeInMinutes - mins) * 60);
    const formattedTime = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;

    const newAnnotation = {
      time: formattedTime,
      event: mark,
      value: 'Пользовательская метка',
    };

    setUserAnnotations((prev) => [...prev, newAnnotation]);
  }, []);

  // Периодический запрос к /analyze каждые 10 секунд при активном WebSocket (только для существующих КТГ)
  useEffect(() => {
    console.log('[HOMEPAGE] useEffect для периодических запросов:', { ktgStatus, isNewKTG });
    if (ktgStatus !== 'recording' || isNewKTG) {
      console.log('[HOMEPAGE] Периодические запросы не запускаются:', { ktgStatus, isNewKTG });
      return;
    }

    console.log('[HOMEPAGE] Запускаем периодические запросы');

    const abortController = new AbortController();

    const fetchAnalysis = async () => {
      try {
        setLoading(true);
        setError(null);

        console.log('[HOMEPAGE] Этап 1: Быстрый запрос основных данных');
        const result = await analysisService.getAnalysisFast(
          actualKtgId || ktgId,
          abortController.signal
        );

        // Проверяем, не был ли запрос отменен
        if (abortController.signal.aborted) {
          console.log('[HOMEPAGE] Запрос анализа отменен при размонтировании');
          return;
        }

        // Сохраняем основные данные в стейт менеджер
        if (result.success && result.data) {
          // Используем updatePartialData чтобы не стереть WebSocket данные
          if (analysisData) {
            // Если уже есть данные, обновляем только новые поля, сохраняя WebSocket данные
            const existingTs = analysisData.ts || [];
            const existingFhr = analysisData.fhr || [];
            const existingToco = analysisData.toco || [];

            updatePartialData({
              ...result.data,
              // Объединяем данные: приоритет у analyze, но сохраняем WebSocket если analyze пустой
              ts: result.data.ts && result.data.ts.length > 0 ? result.data.ts : existingTs,
              fhr: result.data.fhr && result.data.fhr.length > 0 ? result.data.fhr : existingFhr,
              toco:
                result.data.toco && result.data.toco.length > 0 ? result.data.toco : existingToco,
            });
          } else {
            // Если данных еще нет, используем setAnalysisData
            setAnalysisData(result.data);
          }
        }

        // Сразу после быстрого запроса запрашиваем predicts
        console.log('[HOMEPAGE] Этап 2: Запрос predicts и графиков');
        try {
          const predictsResult = await analysisService.getAnalysisPredicts(
            actualKtgId || ktgId,
            abortController.signal
          );

          if (abortController.signal.aborted) {
            console.log('[HOMEPAGE] Запрос predicts отменен при размонтировании');
            return;
          }

          // Обновляем данными predicts
          if (predictsResult.success && predictsResult.data) {
            updatePartialData(predictsResult.data);
          }
        } catch (predictsError) {
          console.warn('[HOMEPAGE] Ошибка загрузки predicts:', predictsError);
          // Не критично, основные данные уже загружены
        }
      } catch (error) {
        // Игнорируем ошибки отмены запроса
        if (error instanceof Error && error.name === 'AbortError') {
          console.log('[HOMEPAGE] Запрос анализа отменен');
          return;
        }

        setError(
          `Ошибка получения анализа: ${
            error instanceof Error ? error.message : 'Неизвестная ошибка'
          }`
        );
      } finally {
        if (!abortController.signal.aborted) {
          setLoading(false);
        }
      }
    };

    // Первый запрос сразу
    fetchAnalysis();

    // Периодические запросы каждые 10 секунд (только быстрый запрос)
    const intervalId = setInterval(async () => {
      try {
        console.log('[HOMEPAGE] Периодический быстрый запрос');
        const result = await analysisService.getAnalysisFast(
          actualKtgId || ktgId,
          abortController.signal
        );

        if (abortController.signal.aborted) {
          return;
        }

        if (result.success && result.data) {
          if (analysisData) {
            const existingTs = analysisData.ts || [];
            const existingFhr = analysisData.fhr || [];
            const existingToco = analysisData.toco || [];

            updatePartialData({
              ...result.data,
              ts: result.data.ts && result.data.ts.length > 0 ? result.data.ts : existingTs,
              fhr: result.data.fhr && result.data.fhr.length > 0 ? result.data.fhr : existingFhr,
              toco:
                result.data.toco && result.data.toco.length > 0 ? result.data.toco : existingToco,
            });
          } else {
            setAnalysisData(result.data);
          }
        }
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          return;
        }
        console.error('[HOMEPAGE] Ошибка периодического запроса:', error);
      }
    }, 10000);

    return () => {
      clearInterval(intervalId);
      abortController.abort();
    };
  }, [ktgStatus, ktgId, actualKtgId, isNewKTG]);
  // Преобразуем данные из анализа в формат для графиков
  const analysisChartData = useMemo(() => {
    if (!analysisData?.ts || !analysisData?.fhr || !analysisData?.toco) {
      return [];
    }

    const { ts, fhr, toco } = analysisData;
    const data: Array<{ ts: number; fhr: number; toco: number | null }> = [];

    // Создаем объекты данных, включая null значения
    for (let i = 0; i < ts.length; i++) {
      data.push({
        ts: ts[i],
        fhr: fhr[i],
        toco: toco[i],
      });
    }

    return data;
  }, [analysisData]);

  // Используем данные из store (которые обновляются и WebSocket'ом и analyze)
  const chartData = analysisChartData;

  // Функция сброса масштаба для обоих графиков
  const handleChartsReset = useCallback(() => {
    console.log('[HOMEPAGE] Сброс масштаба для обоих графиков');
    // Сбрасываем диапазон окна на полный
    if (chartData && chartData.length > 0) {
      const lastPoint = chartData[chartData.length - 1];
      const maxTime = lastPoint ? lastPoint.ts / 60 : 50; // конвертируем в минуты
      const endTime = Math.max(maxTime, 50); // минимум 50 минут
      setWindowRange({ start: 0, end: endTime });
    } else {
      setWindowRange({ start: 0, end: 50 });
    }
  }, [chartData]);

  // Вычисляем baseTs для корректного отображения аннотаций
  const baseTs = useMemo(() => {
    if (!chartData?.length) return null;
    try {
      const timestamps = chartData
        .map((d) => d.ts)
        .filter((ts) => typeof ts === 'number' && !isNaN(ts));
      if (timestamps.length === 0) return null;
      return timestamps.reduce((min, ts) => Math.min(min, ts), timestamps[0]);
    } catch (error) {
      console.error('[HOMEPAGE] Ошибка при вычислении baseTs:', error);
      return null;
    }
  }, [chartData]);

  // Данные для аннотаций из анализа (события)
  const annotations = useMemo(() => {
    const analysisAnnotations = !analysisData?.events
      ? []
      : analysisData.events
          .filter((event) => event.toco_rel !== 'artifact') // исключаем артефакты
          .map((event, index) => {
            // Конвертируем абсолютное время в относительное (от baseTs)
            const absoluteStartTime = event.t_start_s; // секунды от эпохи
            const relativeStartTime = baseTs
              ? (absoluteStartTime - baseTs) / 60
              : event.t_start_s / 60; // минуты от начала записи
            const duration = event.duration_s / 60; // длительность в минутах

            // Форматируем время начала
            const formatTime = (minutes: number) => {
              const mins = Math.floor(minutes);
              const secs = Math.round((minutes - mins) * 60);
              return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            };

            // Определяем название события
            let eventName = '';
            if (event.kind === 'accel') {
              eventName = 'Акцелерация';
            } else if (event.kind === 'decel') {
              if (event.toco_rel === 'early') {
                eventName = 'Ранняя децелерация';
              } else if (event.toco_rel === 'late') {
                eventName = 'Поздняя децелерация';
              } else if (event.toco_rel === 'variable') {
                eventName = 'Вариабельная децелерация';
              } else {
                eventName = 'Децелерация';
              }
            } else if (event.kind === 'tachy') {
              eventName = 'Тахикардия';
            } else if (event.kind === 'brady') {
              eventName = 'Брадикардия';
            }

            return {
              time: formatTime(relativeStartTime),
              event: eventName,
              value: `${duration.toFixed(1)} мин`,
            };
          });

    // Добавляем аннотации из массива stirrings
    const stirringsAnnotations = !analysisData?.stirrings
      ? []
      : analysisData.stirrings.map((prop) => {
          const stirring = prop.additionalProp1;
          // Конвертируем абсолютное время в относительное (от baseTs)
          const absoluteTime = stirring.ts; // секунды от эпохи
          const relativeTime = baseTs ? (absoluteTime - baseTs) / 60 : stirring.ts / 60; // минуты от начала записи
          const mins = Math.floor(relativeTime);
          const secs = Math.round((relativeTime - mins) * 60);
          const formattedTime = `${mins.toString().padStart(2, '0')}:${secs
            .toString()
            .padStart(2, '0')}`;

          return {
            time: formattedTime,
            event: stirring.type || 'Шевеление', // Используем type как текст события
            value: 'Зарегистрировано',
          };
        });

    // Объединяем все типы аннотаций, исключая дублирование
    const allAnnotations = [...analysisAnnotations, ...stirringsAnnotations];

    // Добавляем пользовательские аннотации только если их нет в stirringsAnnotations
    userAnnotations.forEach((userAnnotation) => {
      const isDuplicate = stirringsAnnotations.some(
        (stirringAnnotation) =>
          stirringAnnotation.time === userAnnotation.time &&
          stirringAnnotation.event === userAnnotation.event
      );

      if (!isDuplicate) {
        allAnnotations.push(userAnnotation);
      }
    });

    // Сортируем все аннотации по времени
    return allAnnotations.sort((a, b) => {
      const timeA = a.time.split(':').map(Number);
      const timeB = b.time.split(':').map(Number);
      return timeA[0] * 60 + timeA[1] - (timeB[0] * 60 + timeB[1]);
    });
  }, [analysisData?.events, analysisData?.stirrings, userAnnotations, baseTs]);

  // Данные для рисков
  const risks = [
    {
      title: 'Риск гипоксии',
      percentage: analysisData?.proba_long ? Math.round(Number(analysisData.proba_long) * 100) : 0,
      color: '#FF8D28',
      description: analysisData?.proba_long
        ? `Вероятность гипоксии: ${(Number(analysisData.proba_long) * 100)?.toFixed(1)}%`
        : 'Риск гипоксии пока не определен.',
    },
    {
      title: 'Оперативные роды',
      percentage: 0,
      color: '#FFCC00',
      description: 'Нет устойчивых признаков необходимости оперативных родов.',
    },
    {
      title: 'Слабость родовой деятельности',
      percentage: 0,
      color: '#FF5F57',
      description: 'Нет устойчивых признаков слабости родовой деятельности.',
    },
    {
      title: 'Отслойка плаценты',
      percentage: 0,
      color: '#27C840',
      description: 'Нет устойчивых признаков отслойки плаценты.',
    },
  ];

  // Данные для прогнозов (используем реальные данные из анализа)
  const forecasts = [
    {
      title: 'Краткосрочный прогноз',
      description: analysisData?.proba_short
        ? `Вероятности децелераций (2 метод):\n\n${Object.entries(analysisData.proba_short)
            .map(
              ([time, prob]) =>
                `${time} мин.: ${(prob * 100).toFixed(1)}% (${(
                  (analysisData.proba_short_cnn?.[time] || 0) * 100
                ).toFixed(1)}%)`
            )
            .join('\n')}`
        : 'Пока нет данных для краткосрочного прогноза.',
    },
    {
      title: 'Долгосрочный прогноз',
      description: analysisData?.proba_long
        ? `Вероятность гипоксии: ${(Number(analysisData.proba_long) * 100)?.toFixed(1)}%`
        : 'Пока нет данных для долгосрочного прогноза.',
    },
  ];

  // Данные для карточек показателей (используем реальные данные из анализа)
  const indicators = [
    {
      title: UI_TEXT.FETAL_HR,
      value: (() => {
        // Получаем последнее не null значение ЧСС плода из данных store
        if (analysisData?.fhr && analysisData.fhr.length > 0) {
          const lastValidFhr = analysisData.fhr
            .filter((fhr) => fhr !== null && fhr !== undefined)
            .pop();

          if (lastValidFhr !== undefined && !isNaN(lastValidFhr)) {
            return lastValidFhr.toFixed(1);
          }
        }

        // Fallback на базальную ЧСС или статичные данные
        return analysisData?.baseline && !isNaN(analysisData.baseline)
          ? analysisData.baseline.toFixed(1)
          : CHART_DATA.INDICATORS.FETAL_HR;
      })(),
      unit: UI_TEXT.BPM,
    },
    {
      title: UI_TEXT.MATERNAL_HR,
      value: CHART_DATA.INDICATORS.MATERNAL_HR, // Пока оставляем статичным, если нет в анализе
      unit: UI_TEXT.BPM,
    },
    {
      title: UI_TEXT.BASAL_HR,
      value:
        analysisData?.baseline && !isNaN(analysisData.baseline)
          ? analysisData.baseline.toFixed(1)
          : CHART_DATA.INDICATORS.BASAL_HR,
      unit: UI_TEXT.BPM,
    },
    {
      title: `Вариабельность (${variabilityMetric.toUpperCase()})`,
      value: (() => {
        if (!analysisData) return CHART_DATA.INDICATORS.VARIABILITY;

        const getValue = () => {
          if (variabilityMetric === 'stv') return analysisData.stv;
          if (variabilityMetric === 'ltv') return analysisData.ltv;
          return analysisData.rmssd;
        };

        const value = getValue();
        return value !== null && value !== undefined && !isNaN(value)
          ? value.toFixed(3)
          : CHART_DATA.INDICATORS.VARIABILITY;
      })(),
      unit: variabilityMetric === 'rmssd' ? 'мс' : 'уд/мин',
      onClick: () => {
        // Переключаем между метриками вариабельности
        setVariabilityMetric((prev) => (prev === 'stv' ? 'ltv' : prev === 'ltv' ? 'rmssd' : 'stv'));
      },
    },
    {
      title: UI_TEXT.ACCELERATIONS,
      value: analysisData?.num_accel?.toString() || CHART_DATA.INDICATORS.ACCELERATIONS,
      unit: UI_TEXT.PIECES,
    },
    {
      title: UI_TEXT.DECELERATIONS,
      value: (() => {
        if (!analysisData?.events) {
          return analysisData?.num_decel?.toString() || CHART_DATA.INDICATORS.DECELERATIONS;
        }

        // Подсчитываем децелерации без артефактов
        const decelEvents = analysisData.events.filter(
          (event) => event.kind === 'decel' && event.toco_rel !== 'artifact'
        );

        return decelEvents.length.toString();
      })(),
      unit: UI_TEXT.PIECES,
      breakdown: (() => {
        if (!analysisData?.events) {
          return [
            {
              label: UI_TEXT.EARLY,
              value: CHART_DATA.INDICATORS.DECELERATIONS_BREAKDOWN.EARLY,
              color: '#FBC72B',
            },
            {
              label: UI_TEXT.LATE,
              value: CHART_DATA.INDICATORS.DECELERATIONS_BREAKDOWN.LATE,
              color: '#EF4640',
            },
            {
              label: UI_TEXT.VARIABLE,
              value: CHART_DATA.INDICATORS.DECELERATIONS_BREAKDOWN.VARIABLE,
              color: '#FF8D41',
            },
          ];
        }

        // Подсчитываем децелерации по типам
        const decelEvents = analysisData.events.filter(
          (event) => event.kind === 'decel' && event.toco_rel !== 'artifact'
        );

        const earlyCount = decelEvents.filter((event) => event.toco_rel === 'early').length;
        const lateCount = decelEvents.filter((event) => event.toco_rel === 'late').length;
        const variableCount = decelEvents.filter((event) => event.toco_rel === 'variable').length;

        return [
          {
            label: UI_TEXT.EARLY,
            value: earlyCount,
            color: '#FBC72B',
          },
          {
            label: UI_TEXT.LATE,
            value: lateCount,
            color: '#EF4640',
          },
          {
            label: UI_TEXT.VARIABLE,
            value: variableCount,
            color: '#FF8D41',
          },
        ];
      })(),
    },
  ];

  return (
    <div className={styles.page}>
      <Header
        breadcrumbItems={['Пациенты', `КТГ ID ${actualKtgId || (isNewKTG ? 'Новое' : ktgId)}`]}
      />

      <div className={styles.statusHeader}>
        <StatusHeader
          status={ktgStatus}
          ktgId={actualKtgId || (isNewKTG ? 'Новое КТГ' : ktgId)}
          onStart={handleStartRecording}
          onComplete={handleCompleteRecording}
          onExit={handleExit}
          onUpload={handleUploadCSV}
          isNewRecord={isNewKTG}
        />
      </div>

      <main className={styles.main}>
        <div className={styles.container}>
          {/* Основной контент */}
          <div className={styles.content}>
            <div className={styles.charts}>
              <Chart
                title={UI_TEXT.FETAL_HR}
                type='fetal-hr'
                className={styles.chart}
                xMin={windowRange.start}
                xMax={windowRange.end}
                annotations={annotations}
                realTimeData={chartData}
                isStreaming={isStreaming}
                events={analysisData?.events || []}
                onAddAnnotation={addUserAnnotation}
                isWebSocketConnected={isWebSocketConnected}
                onDoubleClick={handleChartsReset}
                setWindowRange={setWindowRange}
              />

              <Chart
                title={UI_TEXT.UTERINE_TONE}
                type='uterine-tone'
                className={styles.chart}
                xMin={windowRange.start}
                xMax={windowRange.end}
                annotations={annotations}
                realTimeData={chartData}
                isStreaming={isStreaming}
                events={analysisData?.events || []}
                isWebSocketConnected={isWebSocketConnected}
                onDoubleClick={handleChartsReset}
                setWindowRange={setWindowRange}
              />

              <TimeScale
                className={styles.timeScale}
                min={0}
                max={50}
                value={windowRange}
                onChange={setWindowRange}
              />
            </div>

            {/* Блоки рисков и прогнозов */}
            <div className={styles.analysisBlocks}>
              <div className={styles.analysisBlocks__left}>
                <RisksPanel risks={risks} />
              </div>
              <div className={styles.analysisBlocks__right}>
                <ForecastPanel forecasts={forecasts} />
              </div>
            </div>
          </div>

          {/* Боковая панель */}
          <aside className={styles.sidebar}>
            <div className={styles.indicators}>
              {ktgStatus === 'completed' ? (
                // Показываем карточку Фишера вместо ЧСС плода и ЧСС матери для завершенного КТГ
                <FisherCard
                  score={analysisData?.fisher_points ?? null}
                  status={(() => {
                    const score = analysisData?.fisher_points;
                    if (score === null || score === undefined) return 'unknown';
                    if (score >= 8) return 'good';
                    if (score >= 6) return 'suspicious';
                    return 'pathological';
                  })()}
                  description={(() => {
                    const score = analysisData?.fisher_points;
                    if (score === null || score === undefined) {
                      return 'Оценка по Фишеру пока не определена. Анализ данных в процессе.';
                    }

                    if (score >= 8) {
                      return 'Отличные показатели КТГ. Все параметры в норме. Продолжайте наблюдение.';
                    } else if (score >= 6) {
                      return 'Показатели требуют внимания. Рекомендуется дополнительное наблюдение и контроль.';
                    } else {
                      return 'Критические показатели КТГ. Требуется немедленное медицинское вмешательство.';
                    }
                  })()}
                />
              ) : (
                // Показываем карточку ЧСС плода для активного КТГ
                <div className={styles.indicators__row}>
                  <IndicatorCard {...indicators[0]} />
                  <IndicatorCard {...indicators[1]} />
                </div>
              )}

              {/* Остальные карточки всегда показываем */}
              <div className={styles.indicators__row}>
                <IndicatorCard {...indicators[2]} />
                <IndicatorCard {...indicators[3]} />
              </div>

              <div className={styles.indicators__row}>
                <IndicatorCard {...indicators[4]} />
                <IndicatorCard {...indicators[5]} />
              </div>
            </div>

            <AnnotationsPanel annotations={annotations} className={styles.annotations} />
          </aside>
        </div>
      </main>
    </div>
  );
};
