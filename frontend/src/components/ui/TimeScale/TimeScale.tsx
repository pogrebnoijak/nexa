import React from 'react';
import { ChevronLeftIcon, ChevronRightIcon } from '../../../icons';
import styles from './TimeScale.module.scss';

export interface TimeScaleProps {
  className?: string;
  min?: number;
  max?: number;
  value: { start: number; end: number };
  onChange: (next: { start: number; end: number }) => void;
}

export const TimeScale: React.FC<TimeScaleProps> = ({
  className = '',
  min = 0,
  max = 50,
  value,
  onChange,
}) => {
  const scaleClasses = [styles.scale, className].filter(Boolean).join(' ');
  const trackRef = React.useRef<HTMLDivElement | null>(null);
  const draggingRef = React.useRef<null | 'start' | 'end' | 'range'>(null);
  const HIT_SLOP_PX = 4;
  const STEP = 0.1;

  const clamp = (n: number, lo: number, hi: number) => Math.min(Math.max(n, lo), hi);
  const pxToTime = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return value.start;
    const rect = el.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
    const time = min + ratio * (max - min);
    return Number((Math.round(time / STEP) * STEP).toFixed(1));
  };

  const moveRange = (direction: 'left' | 'right') => {
    const rangeWidth = value.end - value.start;
    const moveStep = 5; // Увеличенный шаг движения в единицах времени

    if (direction === 'left') {
      const newStart = Math.max(min, value.start - moveStep);
      const newEnd = newStart + rangeWidth;
      onChange({ start: newStart, end: newEnd });
    } else {
      const newEnd = Math.min(max, value.end + moveStep);
      const newStart = newEnd - rangeWidth;
      onChange({ start: newStart, end: newEnd });
    }
  };

  // Проверяем, можно ли двигаться влево
  const canMoveLeft = value.start > min;
  // Проверяем, можно ли двигаться вправо
  const canMoveRight = value.end < max;

  const onMouseDown: React.MouseEventHandler<HTMLDivElement> = (e) => {
    const el = trackRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const startPx = ((value.start - min) / (max - min)) * rect.width + rect.left;
    const endPx = ((value.end - min) / (max - min)) * rect.width + rect.left;
    const x = e.clientX;

    const nearStart = Math.abs(x - startPx) <= HIT_SLOP_PX;
    const nearEnd = Math.abs(x - endPx) <= HIT_SLOP_PX;
    const inRange = x >= startPx && x <= endPx;

    if (nearStart) {
      draggingRef.current = 'start';
    } else if (nearEnd) {
      draggingRef.current = 'end';
    } else if (inRange) {
      draggingRef.current = 'range';
    } else {
      return;
    }

    const initialX = e.clientX;
    const initialStart = value.start;
    const initialEnd = value.end;

    const move = (ev: MouseEvent) => {
      const deltaX = ev.clientX - initialX;
      const deltaTime = (deltaX / rect.width) * (max - min);

      if (draggingRef.current === 'start') {
        const t = pxToTime(ev.clientX);
        const nextStart = Math.min(t, value.end - STEP);
        onChange({ start: nextStart, end: value.end });
      } else if (draggingRef.current === 'end') {
        const t = pxToTime(ev.clientX);
        const nextEnd = Math.max(t, value.start + STEP);
        onChange({ start: value.start, end: nextEnd });
      } else if (draggingRef.current === 'range') {
        const newStart = Math.max(
          min,
          Math.min(max - (initialEnd - initialStart), initialStart + deltaTime)
        );
        const newEnd = Math.min(
          max,
          Math.max(min + (initialEnd - initialStart), initialEnd + deltaTime)
        );
        onChange({ start: newStart, end: newEnd });
      }
    };

    const up = () => {
      draggingRef.current = null;
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };

    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };

  return (
    <div className={scaleClasses}>
      <button
        className={styles.scale__button}
        onClick={() => moveRange('left')}
        disabled={!canMoveLeft}
      >
        <ChevronLeftIcon />
      </button>

      <div className={styles.scale__track} ref={trackRef} onMouseDown={onMouseDown}>
        <div
          className={styles.scale__range}
          style={{
            left: `${((value.start - min) / (max - min)) * 100}%`,
            width: `${((value.end - value.start) / (max - min)) * 100}%`,
          }}
        />
        <div
          className={styles.scale__handle}
          style={{ left: `${((value.start - min) / (max - min)) * 100}%` }}
        />
        <div
          className={styles.scale__handle}
          style={{ left: `${((value.end - min) / (max - min)) * 100}%` }}
        />
      </div>

      <button
        className={`${styles.scale__button} ${styles['scale__button--right']}`}
        onClick={() => moveRange('right')}
        disabled={!canMoveRight}
      >
        <ChevronRightIcon />
      </button>
    </div>
  );
};
