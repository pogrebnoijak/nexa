import React, { useState, useRef, useCallback, useEffect, forwardRef } from 'react';
import { BurgerIcon, CloseIcon } from '../../../icons';
import styles from './SidebarHandle.module.scss';

interface SidebarHandleProps {
  isOpen: boolean;
  onToggle: (isOpen: boolean) => void;
  className?: string;
  isVisible?: boolean;
}

const SidebarHandle = forwardRef<HTMLDivElement, SidebarHandleProps>(
  ({ isOpen, onToggle, className = '', isVisible = true }, ref) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isMouseDown, setIsMouseDown] = useState(false);
    const [position, setPosition] = useState(() => {
      const saved = localStorage.getItem('sidebar-handle-position');
      const defaultPos = 382;
      const minPos = 24;
      const maxPos = window.innerHeight - 100 - 24;
      const savedPos = saved ? parseInt(saved, 10) : defaultPos;
      return Math.max(minPos, Math.min(maxPos, savedPos));
    });

    const startYRef = useRef<number>(0);
    const startPositionRef = useRef<number>(0);

    // Сохраняем позицию в localStorage
    useEffect(() => {
      localStorage.setItem('sidebar-handle-position', position.toString());
    }, [position]);

    const handleMouseDown = useCallback(
      (e: React.MouseEvent) => {
        e.preventDefault();
        setIsMouseDown(true);
        startYRef.current = e.clientY;
        startPositionRef.current = position;
      },
      [position]
    );

    const handleMouseMove = useCallback(
      (e: MouseEvent) => {
        if (!isMouseDown) return;

        const deltaY = Math.abs(e.clientY - startYRef.current);

        // Если мышь сдвинулась больше чем на 5px, считаем это перетаскиванием
        if (deltaY > 5 && !isDragging) {
          setIsDragging(true);
        }

        if (!isDragging) return;

        const actualDeltaY = e.clientY - startYRef.current;
        const newPosition = Math.max(
          24, // Минимум 24px от верха (область скругления сайдбара)
          Math.min(window.innerHeight - 100 - 24, startPositionRef.current + actualDeltaY) // Минимум 24px от низа
        );
        setPosition(newPosition);
      },
      [isDragging, isMouseDown]
    );

    const handleMouseUp = useCallback(() => {
      setIsMouseDown(false);
      // Небольшая задержка перед сбросом isDragging, чтобы handleClick мог его проверить
      setTimeout(() => {
        setIsDragging(false);
      }, 10);
    }, []);

    // Обработчики для перетаскивания
    useEffect(() => {
      if (isMouseDown) {
        const handleMouseMoveGlobal = (e: MouseEvent) => {
          handleMouseMove(e);
        };

        const handleMouseUpGlobal = () => {
          handleMouseUp();
        };

        document.addEventListener('mousemove', handleMouseMoveGlobal);
        document.addEventListener('mouseup', handleMouseUpGlobal);

        if (isDragging) {
          document.body.style.userSelect = 'none';
        }

        return () => {
          document.removeEventListener('mousemove', handleMouseMoveGlobal);
          document.removeEventListener('mouseup', handleMouseUpGlobal);
          document.body.style.userSelect = '';
        };
      }
    }, [isMouseDown, isDragging, handleMouseMove, handleMouseUp]);

    const handleClick = useCallback(
      (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();

        // Если это был клик (не перетаскивание), переключаем меню
        // Проверяем, что мышь не двигалась значительно от начальной позиции
        const deltaY = Math.abs(e.clientY - startYRef.current);
        if (deltaY <= 5) {
          onToggle(!isOpen);
        }
      },
      [isOpen, onToggle]
    );

    // Автоматически определяем состояние ручки в зависимости от состояния меню
    const getHandleState = () => {
      if (!isVisible) return 'hidden';
      if (isDragging && isOpen) return 'pressed-opened';
      if (isDragging && !isOpen) return 'pressed-closed';
      if (isOpen) return 'opened';
      return 'default';
    };

    return (
      <div
        ref={ref}
        className={`${styles.handle} ${className}`}
        style={{ top: `${position}px` }}
        onMouseDown={handleMouseDown}
        onClick={handleClick}
        data-state={getHandleState()}
      >
        <div className={styles.handle__icon}>
          {isOpen ? (
            <CloseIcon className={styles.handle__svg} />
          ) : (
            <BurgerIcon className={`${styles.handle__svg} ${styles['handle__svg--burger']}`} />
          )}
        </div>
      </div>
    );
  }
);

SidebarHandle.displayName = 'SidebarHandle';

export default SidebarHandle;
