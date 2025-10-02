import React, { useMemo, useRef, useEffect } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { toast } from 'sonner';
import { CTGStreamData } from '../../../api/types';
import { Event } from '../../../store/types';
import { MarkSelector } from '../../ui/MarkSelector';
import styles from './Chart.module.scss';

export interface Annotation {
  time: string;
  event: string;
  value: string;
}

export interface ChartProps {
  title: string;
  type: 'fetal-hr' | 'uterine-tone';
  className?: string;
  xMin?: number;
  xMax?: number;
  annotations?: Annotation[];
  realTimeData?: CTGStreamData[];
  isStreaming?: boolean;
  events?: Event[];
  onAddAnnotation?: (mark: string, timestamp: number) => void;
  isWebSocketConnected?: boolean;
  onDoubleClick?: () => void;
  setWindowRange?: (range: { start: number; end: number }) => void;
}

// Плагин для тултипа
function tooltipPlugin(getData: () => CTGStreamData[], isFetalHr: boolean, baseTs: number | null) {
  let tooltip: HTMLDivElement | null = null;

  return {
    hooks: {
      init: (u: uPlot) => {
        // Создаем элемент тултипа
        tooltip = document.createElement('div');
        tooltip.className = 'chart-tooltip';
        tooltip.style.cssText = `
          position: absolute;
          background: rgba(0, 0, 0, 0.8);
          color: white;
          padding: 8px 12px;
          border-radius: 4px;
          font-size: 12px;
          font-family: Inter, sans-serif;
          pointer-events: none;
          z-index: 1000;
          display: none;
          white-space: nowrap;
        `;
        u.root.appendChild(tooltip);
      },
      setCursor: (u: uPlot) => {
        if (!tooltip) return;

        const { left, top, idx } = u.cursor;

        if (
          left === null ||
          top === null ||
          idx === null ||
          left === undefined ||
          top === undefined
        ) {
          tooltip.style.display = 'none';
          return;
        }

        // Проверяем baseTs
        if (baseTs == null) {
          tooltip.style.display = 'none';
          return;
        }

        // Находим ближайшую точку данных (relative -> absolute)
        const relMin = Math.max(0, u.posToVal(left, 'x')); // минуты от base, с clamp
        const absSec = baseTs + relMin * 60;

        // Получаем свежие данные и ищем ближайшую точку по absolute времени
        const realTimeData = getData();
        let closestPoint: CTGStreamData | null = null;
        let minDistance = Infinity;

        for (const point of realTimeData) {
          const distance = Math.abs(point.ts - absSec);
          if (distance < minDistance) {
            minDistance = distance;
            closestPoint = point;
          }
        }

        if (closestPoint && minDistance < 30) {
          // показываем тултип только если точка близко (в пределах 30 секунд)
          const fhrValue = closestPoint.fhr;
          const tocoValue = closestPoint.toco;

          // Проверяем, есть ли данные для отображения (не показываем для null значений)
          const currentValue = isFetalHr ? fhrValue : tocoValue;
          if (currentValue === null) {
            tooltip.style.display = 'none';
            return;
          }

          let content = '';
          if (isFetalHr) {
            content = `ЧСС плода: ${fhrValue!.toFixed(0)} уд/мин`;
            if (tocoValue !== null) {
              content += `\nТонус матки: ${tocoValue.toFixed(1)} мм рт.ст.`;
            }
          } else {
            content = `Тонус матки: ${tocoValue!.toFixed(1)} мм рт.ст.`;
            if (fhrValue !== null) {
              content += `\nЧСС плода: ${fhrValue.toFixed(0)} уд/мин`;
            }
          }

          const time = new Date(closestPoint.ts * 1000).toISOString().substr(14, 5); // MM:SS
          content += `\nВремя: ${time}`;

          tooltip.innerHTML = content.replace(/\n/g, '<br>');
          tooltip.style.display = 'block';

          // Позиционируем тултип относительно точки данных по relative X
          const pointTimeInMinutes = (closestPoint.ts - baseTs) / 60; // relative время
          const pointValue = currentValue;

          // Конвертируем координаты точки в пиксели
          const pointX = u.valToPos(pointTimeInMinutes, 'x') + 56;
          const pointY = u.valToPos(pointValue, 'y') + 12;

          // Позиционируем тултип так, чтобы левый нижний угол совпадал с точкой
          const tooltipRect = tooltip.getBoundingClientRect();

          tooltip.style.left = pointX + 'px';
          tooltip.style.top = pointY - tooltipRect.height + 'px';

          // Проверяем, не выходит ли тултип за границы
          const updatedTooltipRect = tooltip.getBoundingClientRect();
          const chartRect = u.root.getBoundingClientRect();

          if (updatedTooltipRect.right > window.innerWidth) {
            tooltip.style.left = pointX - updatedTooltipRect.width + 'px';
          }
          if (updatedTooltipRect.top < chartRect.top) {
            tooltip.style.top = pointY + 'px';
          }
        } else {
          tooltip.style.display = 'none';
        }
      },
      destroy: (u: uPlot) => {
        if (tooltip) {
          tooltip.remove();
          tooltip = null;
        }
      },
    },
  };
}

// Плагин для отрисовки цветных зон с границами
function zonesPlugin(ranges: Array<[number, number, string, string]>) {
  return {
    hooks: {
      drawClear: (u: uPlot) => {
        const { ctx } = u;
        if (!ctx) return;

        ranges.forEach(([x0, x1, rgba, strokeColor]) => {
          const x0px = u.valToPos(x0, 'x', true);
          const x1px = u.valToPos(x1, 'x', true);
          if (x0px === null || x1px === null) return;

          // Рисуем заливку
          ctx.fillStyle = rgba;
          ctx.fillRect(x0px, u.bbox.top, x1px - x0px, u.bbox.height);

          // Рисуем вертикальные прерывистые линии по краям
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 2;
          ctx.setLineDash([10, 10]); // прерывистая линия

          // Левая граница
          ctx.beginPath();
          ctx.moveTo(x0px, u.bbox.top);
          ctx.lineTo(x0px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          // Правая граница
          ctx.beginPath();
          ctx.moveTo(x1px, u.bbox.top);
          ctx.lineTo(x1px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          ctx.setLineDash([]); // сбрасываем
        });
      },
    },
  };
}

// Плагин для отрисовки событий
function eventsPlugin(getEvents: () => Event[], isFetalHr: boolean, baseTs: number | null) {
  return {
    hooks: {
      drawClear: (u: uPlot) => {
        const { ctx } = u;
        const events = getEvents();
        if (!ctx || !events || events.length === 0 || baseTs == null) return;

        // Фильтруем события (исключаем artifact)
        const filteredEvents = events.filter((event) => event.toco_rel !== 'artifact');

        filteredEvents.forEach((event) => {
          const startTime = (event.t_start_s - baseTs) / 60; // relative время начала в минутах
          const endTime = (event.t_start_s + event.duration_s - baseTs) / 60; // relative время окончания в минутах

          const x0px = u.valToPos(startTime, 'x', true);
          const x1px = u.valToPos(endTime, 'x', true);

          if (x0px === null || x1px === null) return;

          // Определяем цвет в зависимости от типа события
          let color = '#000000';
          if (event.kind === 'accel') {
            color = '#27C840'; // зеленый
          } else if (event.kind === 'decel') {
            if (event.toco_rel === 'early') {
              color = '#FFCC00'; // желтый
            } else if (event.toco_rel === 'late') {
              color = '#FF5F57'; // красный
            } else if (event.toco_rel === 'variable') {
              color = '#FF8D28'; // оранжевый
            }
          }

          // Рисуем заливку события
          ctx.fillStyle = color + '20'; // добавляем прозрачность
          ctx.fillRect(x0px, u.bbox.top, x1px - x0px, u.bbox.height);

          // Рисуем границы события
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]); // пунктирная линия

          // Левая граница
          ctx.beginPath();
          ctx.moveTo(x0px, u.bbox.top);
          ctx.lineTo(x0px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          // Правая граница
          ctx.beginPath();
          ctx.moveTo(x1px, u.bbox.top);
          ctx.lineTo(x1px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          ctx.setLineDash([]); // сбрасываем
        });
      },
    },
  };
}

// Плагин для отрисовки только границ событий (без заливки) - для второго графика
function eventsBordersPlugin(getEvents: () => Event[], baseTs: number | null) {
  return {
    hooks: {
      drawClear: (u: uPlot) => {
        const { ctx } = u;
        const events = getEvents();
        if (!ctx || !events || events.length === 0 || baseTs == null) return;

        // Фильтруем события (исключаем artifact)
        const filteredEvents = events.filter((event) => event.toco_rel !== 'artifact');

        filteredEvents.forEach((event) => {
          const startTime = (event.t_start_s - baseTs) / 60; // relative время начала в минутах
          const endTime = (event.t_start_s + event.duration_s - baseTs) / 60; // relative время окончания в минутах

          const x0px = u.valToPos(startTime, 'x', true);
          const x1px = u.valToPos(endTime, 'x', true);

          if (x0px === null || x1px === null) return;

          // Определяем цвет в зависимости от типа события
          let color = '#000000';
          if (event.kind === 'accel') {
            color = '#27C840'; // зеленый
          } else if (event.kind === 'decel') {
            if (event.toco_rel === 'early') {
              color = '#FFCC00'; // желтый
            } else if (event.toco_rel === 'late') {
              color = '#FF5F57'; // красный
            } else if (event.toco_rel === 'variable') {
              color = '#FF8D28'; // оранжевый
            }
          }

          // Рисуем только границы события (без заливки)
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]); // пунктирная линия

          // Левая граница
          ctx.beginPath();
          ctx.moveTo(x0px, u.bbox.top);
          ctx.lineTo(x0px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          // Правая граница
          ctx.beginPath();
          ctx.moveTo(x1px, u.bbox.top);
          ctx.lineTo(x1px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          ctx.setLineDash([]); // сбрасываем
        });
      },
    },
  };
}

// Плагин для отрисовки только границ зон (без заливки)
function zoneBordersPlugin(ranges: Array<[number, number, string]>) {
  return {
    hooks: {
      drawClear: (u: uPlot) => {
        const { ctx } = u;
        if (!ctx) return;

        ranges.forEach(([x0, x1, strokeColor]) => {
          const x0px = u.valToPos(x0, 'x', true);
          const x1px = u.valToPos(x1, 'x', true);
          if (x0px === null || x1px === null) return;

          // Рисуем только вертикальные прерывистые линии по краям
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 2;
          ctx.setLineDash([10, 10]); // прерывистая линия

          // Левая граница
          ctx.beginPath();
          ctx.moveTo(x0px, u.bbox.top);
          ctx.lineTo(x0px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          // Правая граница
          ctx.beginPath();
          ctx.moveTo(x1px, u.bbox.top);
          ctx.lineTo(x1px, u.bbox.top + u.bbox.height);
          ctx.stroke();

          ctx.setLineDash([]); // сбрасываем
        });
      },
    },
  };
}

// Плагин для отрисовки горизонтальных зон (например, нормальные значения ЧСС)
function horizontalZonesPlugin(zones: Array<[number, number, string]>, isFetalHr: boolean) {
  return {
    hooks: {
      drawClear: (u: uPlot) => {
        const { ctx } = u;
        if (!ctx || !isFetalHr) return; // Показываем только для ЧСС

        zones.forEach(([y0, y1, fillColor]) => {
          const y0px = u.valToPos(y0, 'y', true);
          const y1px = u.valToPos(y1, 'y', true);
          if (y0px === null || y1px === null) return;

          // Рисуем горизонтальную зону на всю ширину графика
          ctx.fillStyle = fillColor;
          ctx.fillRect(u.bbox.left, y1px, u.bbox.width, y0px - y1px);
        });
      },
    },
  };
}

// Плагин для отрисовки аннотаций (черные вертикальные линии)
function annotationsPlugin(
  getAnnotations: () => Annotation[],
  isFetalHr: boolean,
  baseTs: number | null
) {
  return {
    hooks: {
      draw: (u: uPlot) => {
        const { ctx } = u;
        if (!ctx) return;

        // Устанавливаем область обрезки по реальному canvas (device pixels)
        ctx.save();
        ctx.beginPath();
        // ✅ Клип по реальному canvas (device pixels), а не CSS-пикселям
        ctx.rect(0, 0, u.ctx.canvas.width, u.ctx.canvas.height);
        ctx.clip();

        const canvasW = u.ctx.canvas.width; // ✅ используем реальную ширину canvas
        const annotations = getAnnotations(); // получаем актуальные аннотации

        annotations.forEach((annotation) => {
          // Парсим время в формате "MM:SS" и конвертируем в минуты с секундами
          const [mm, ss] = annotation.time.split(':').map(Number);

          // Время аннотации в минутах (это уже относительное время от начала записи)
          const timeValue = mm + ss / 60;

          const xPx = u.valToPos(timeValue, 'x', true);
          if (xPx == null) {
            return;
          }

          // Проверяем, что аннотация находится в видимой области canvas
          const rectWidth = 7;
          const rectX = xPx - rectWidth / 2; // центрируем по x

          // ✅ Проверка границ по canvas, а не по u.over
          if (rectX + rectWidth < 0 || rectX > canvasW) {
            return;
          }

          // Рисуем черную вертикальную линию от низа области данных до самого низа графика
          ctx.fillStyle = '#000000';

          const rectY = u.bbox.top + u.bbox.height - 36;
          const rectHeight = 36;

          ctx.fillRect(rectX, rectY, rectWidth, rectHeight);
        });

        // Восстанавливаем контекст
        ctx.restore();
      },
    },
  };
}

export const Chart: React.FC<ChartProps> = ({
  title,
  type,
  className = '',
  xMin,
  xMax,
  annotations = [],
  realTimeData = [],
  isStreaming = false,
  events = [],
  onAddAnnotation,
  isWebSocketConnected = false,
  onDoubleClick,
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const plotInstance = useRef<uPlot | null>(null);
  const isUserLocked = useRef(false);

  // Refs для данных, чтобы не пересоздавать график
  const dataRef = useRef<CTGStreamData[]>([]);
  const eventsRef = useRef<Event[]>([]);
  const annotationsRef = useRef<Annotation[]>([]);

  // Обновляем refs при изменении данных
  useEffect(() => {
    dataRef.current = realTimeData ?? [];
  }, [realTimeData]);
  useEffect(() => {
    eventsRef.current = events ?? [];
  }, [events]);
  useEffect(() => {
    annotationsRef.current = annotations ?? [];
  }, [annotations]);

  const chartClasses = [styles.chart, className].filter(Boolean).join(' ');
  const isFetalHr = type === 'fetal-hr';
  const chartHeight = useMemo(() => (isFetalHr ? 314 : 243), [isFetalHr]);

  // Единый baseTs для всех плагинов
  const baseTs = useMemo(() => {
    if (!realTimeData?.length) return null;
    try {
      // Ограничиваем количество элементов для вычисления baseTs
      const maxElementsForBaseTs = 10000;
      const dataForBaseTs =
        realTimeData.length > maxElementsForBaseTs
          ? realTimeData.slice(0, maxElementsForBaseTs) // берем первые элементы для baseTs
          : realTimeData;

      const timestamps = dataForBaseTs
        .map((d) => d.ts)
        .filter((ts) => typeof ts === 'number' && !isNaN(ts));
      if (timestamps.length === 0) return null;

      // Используем reduce вместо Math.min(...array) для избежания переполнения стека
      return timestamps.reduce((min, ts) => Math.min(min, ts), timestamps[0]);
    } catch (error) {
      console.error('[CHART] Ошибка при вычислении baseTs:', error);
      return null;
    }
  }, [realTimeData]);

  // Сбрасываем лок при смене baseTs (новая сессия/переподключение)
  useEffect(() => {
    if (baseTs == null) return;
    isUserLocked.current = false;
  }, [baseTs]);

  // Подготовка данных: X = (ts - baseTs) / 60, разрывы = null
  const chartData = useMemo((): uPlot.AlignedData => {
    if (!realTimeData?.length || baseTs == null) return [[], []];

    try {
      // Данные гарантированно отсортированы, поэтому просто обрабатываем их
      const x: number[] = [];
      const y: (number | null)[] = [];

      for (let i = 0; i < realTimeData.length; i++) {
        const p = realTimeData[i];

        // Минимальная валидация
        if (!p || typeof p.ts !== 'number' || isNaN(p.ts)) continue;

        const val = isFetalHr ? p.fhr : p.toco;

        x.push((p.ts - baseTs) / 60); // относительные минуты
        y.push(val == null ? null : val);

        // Проверяем разрывы только если нужно
        if (i < realTimeData.length - 1) {
          const n = realTimeData[i + 1];
          const nextVal = isFetalHr ? n.fhr : n.toco;
          if (nextVal == null && val != null) {
            const mid = ((p.ts + n.ts) / 2 - baseTs) / 60;
            x.push(mid);
            y.push(null);
          }
        }
      }
      return [x, y];
    } catch (error) {
      console.error('[CHART] Ошибка при подготовке данных:', error);
      return [[], []];
    }
  }, [isFetalHr, realTimeData, baseTs]);

  // Настройки uPlot
  const options = useMemo<uPlot.Options>(() => {
    const hasHardRange = typeof xMin === 'number' || typeof xMax === 'number';

    return {
      width: 800,
      height: chartHeight,
      pxAlign: 0.5,
      series: [
        {}, // x axis
        {
          stroke: '#000000',
          width: 1.25,
          points: { show: false },
          spanGaps: false,
        },
      ],
      axes: [
        {
          values: (u, vals) => vals.map((v) => v.toFixed(1)),
        },
        {
          scale: 'y',
        },
      ],
      scales: {
        x: {
          auto: false,
          // ⚠️ никаких initialX здесь!
          ...(hasHardRange ? { min: xMin, max: xMax } : {}),
        },
        y: {
          auto: false,
          range: isFetalHr ? [60, 220] : [0, 100],
          // Принудительно фиксируем диапазон для ЧСС
          ...(isFetalHr && {
            min: 60,
            max: 220,
          }),
          // Устанавливаем интервал тиков: каждые 20 для ЧСС, каждые 25 для тонуса
          ...(isFetalHr ? { incrs: [20] } : { incrs: [25] }),
        },
      },
      cursor: {
        drag: {
          x: true, // Разрешаем перетаскивание по оси X для приближения
          y: false,
        },
      },
      hooks: {
        setSelect: [
          (u: uPlot) => {
            const sel = u.select;
            // Если реально есть выделение по X → считаем, что юзер зафиксировал масштаб
            if (sel.width > 0) isUserLocked.current = true;
          },
        ],
      },
      legend: {
        show: false,
      },
      plugins: [
        // Горизонтальная зона нормальных значений ЧСС (120-160 bpm)
        ...(isFetalHr ? [horizontalZonesPlugin([[120, 160, '#E9ECEC']], isFetalHr)] : []),
        // Тултип для отображения значений ЧСС и тонуса матки
        tooltipPlugin(() => dataRef.current, isFetalHr, baseTs),
        // Для ЧСС - события с заливкой, для Тонуса - только границы событий
        ...(isFetalHr ? [eventsPlugin(() => eventsRef.current, isFetalHr, baseTs)] : []),
        ...(!isFetalHr ? [eventsBordersPlugin(() => eventsRef.current, baseTs)] : []),
        annotationsPlugin(() => annotationsRef.current, isFetalHr, baseTs),
      ],
    };
  }, [isFetalHr, xMin, xMax, chartHeight, baseTs]); // ⬅️ нет chartData/isStreaming/annotations

  // Инициализация графика (без chartData в зависимостях!)
  useEffect(() => {
    if (!chartRef.current) return;

    if (plotInstance.current) plotInstance.current.destroy();
    plotInstance.current = new uPlot(options, chartData, chartRef.current);

    const u = plotInstance.current;

    if (isFetalHr) u.setScale('y', { min: 60, max: 220 });

    // Устанавливаем дефолтный диапазон X, если не заданы внешние ограничения
    if (typeof xMin !== 'number' && typeof xMax !== 'number') {
      u.setScale('x', { min: 0, max: 50 });
      console.log('set 1');
    }

    const fit = () => {
      if (!chartRef.current || !u) return;
      console.log('fit');
      u.setSize({ width: chartRef.current.offsetWidth, height: chartHeight });
    };

    requestAnimationFrame(fit);

    const ro = new ResizeObserver(fit);
    chartRef.current.parentElement && ro.observe(chartRef.current.parentElement);

    return () => ro.disconnect();
  }, [options, chartHeight, isFetalHr]);

  // Двойной клик для сброса лока масштаба
  useEffect(() => {
    const u = plotInstance.current;
    if (!u) return;

    const onDblClick = () => {
      if (onDoubleClick) {
        // Вызываем внешний обработчик, который сбросит оба графика
        onDoubleClick();
      } else {
        // Fallback: сбрасываем только текущий график
        isUserLocked.current = false; // снимаем лок
        // сбрасываем селект и возвращаемся к авто-диапазону по данным
        u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
        // правую границу выставим сразу по данным:
        const xs = chartData[0] as number[];
        if (xs.length) {
          const dataMax = xs[xs.length - 1];
          const newMax = dataMax;
          const newMin = 0;
          console.log('set 3');
          u.setScale('x', { min: newMin, max: newMax });
        }
      }
    };

    u.over.addEventListener('dblclick', onDblClick);
    return () => u.over.removeEventListener('dblclick', onDblClick);
  }, [chartData, isStreaming, onDoubleClick]);

  // Обновление данных без пересоздания графика
  useEffect(() => {
    plotInstance.current?.setData(chartData);
  }, [chartData]);

  // Управляем X ТОЛЬКО здесь, и только если масштаб не залочен
  // Live-обновление границ отключено
  useEffect(() => {
    const u = plotInstance.current;
    if (!u || isUserLocked.current) return;
    const xs = chartData[0] as number[];
    if (!xs?.length) return;
    const last = xs[xs.length - 1];

    console.log('set 4' + last);
    u.setScale('x', { min: 0, max: last });
  }, [chartData]);

  // Очистка при размонтировании
  useEffect(() => {
    return () => {
      if (plotInstance.current) {
        plotInstance.current.destroy();
        plotInstance.current = null;
      }
    };
  }, []);

  const handleMarkSelect = async (mark: string, timestamp: number) => {
    // Отправляем на сервер
    try {
      const response = await fetch('http://localhost:8000/stirring', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          additionalProp1: {
            type: mark,
            ts: timestamp,
          },
        }),
      });

      if (response.ok) {
        // Успешно отправлено - добавляем в UI
        if (onAddAnnotation) {
          onAddAnnotation(mark, timestamp);
        }
        toast.success(`Метка "${mark}" добавлена`);
      } else {
        // Обрабатываем ошибку от сервера
        const errorData = await response.json().catch(() => ({ error: 'Неизвестная ошибка' }));
        toast.error(errorData.error || 'Ошибка при добавлении метки');
      }
    } catch (error) {
      // Ошибка сети или другая
      toast.error('Ошибка соединения с сервером');
    }
  };

  return (
    <div className={chartClasses}>
      <div className={styles.chart__header}>
        <h3 className={styles.chart__title}>{title}</h3>
        <div className={styles.chart__headerRight}>
          {isFetalHr && isWebSocketConnected && (
            <MarkSelector
              onMarkSelect={handleMarkSelect}
              lastTimestamp={
                realTimeData && realTimeData.length > 0
                  ? realTimeData[realTimeData.length - 1].ts
                  : 0
              }
              className={styles.chart__markSelector}
            />
          )}
          {isStreaming && <span className={styles.chart__streaming}>● LIVE</span>}
        </div>
      </div>

      <div className={styles.chart__container} style={{ height: chartHeight, overflow: 'hidden' }}>
        <div ref={chartRef} style={{ width: '100%', height: '100%' }} />
      </div>
    </div>
  );
};
