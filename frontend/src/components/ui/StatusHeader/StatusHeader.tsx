import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../Button';
import { ArrowLeftIcon } from '../../../icons';
import styles from './StatusHeader.module.scss';

export interface StatusHeaderProps {
  status: 'idle' | 'recording' | 'completed' | 'loading';
  ktgId: string;
  recordingTime?: string; // для состояния "Идет запись"
  onStart?: () => void; // для кнопки "Начать запись"
  onComplete?: () => void; // для кнопки "Завершить"
  onExit?: () => void; // для кнопки "Назад" - сброс состояния
  onUpload?: (file: File) => void; // для кнопки "Загрузить CSV"
  isNewRecord?: boolean; // флаг новой записи
}

export const StatusHeader: React.FC<StatusHeaderProps> = ({
  status,
  ktgId,
  recordingTime,
  onStart,
  onComplete,
  onExit,
  onUpload,
  isNewRecord = false,
}) => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleBackClick = () => {
    // Вызываем сброс состояния если передан обработчик
    if (onExit) {
      onExit();
    }
    navigate('/');
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv' && onUpload) {
      onUpload(file);
    } else if (file && file.type !== 'text/csv') {
      alert('Пожалуйста, выберите CSV файл');
    }
    // Сбрасываем значение input для возможности повторной загрузки того же файла
    if (event.target) {
      event.target.value = '';
    }
  };

  const [currentTime, setCurrentTime] = useState('00:00');

  // Таймер для состояния "Идет запись"
  useEffect(() => {
    if (status === 'recording') {
      setCurrentTime('00:00'); // Сброс при старте
      const interval = setInterval(() => {
        setCurrentTime((prev) => {
          const [minutes, seconds] = prev.split(':').map(Number);
          const totalSeconds = minutes * 60 + seconds + 1;
          const newMinutes = Math.floor(totalSeconds / 60);
          const newSeconds = totalSeconds % 60;
          return `${newMinutes.toString().padStart(2, '0')}:${newSeconds
            .toString()
            .padStart(2, '0')}`;
        });
      }, 1000);

      return () => {
        clearInterval(interval);
      };
    } else {
    }
  }, [status]);

  return (
    <div className={styles.statusHeader}>
      <div className={styles.statusHeader__content}>
        <div className={styles.statusHeader__left}>
          <div className={styles.statusHeader__icon} onClick={handleBackClick}>
            <ArrowLeftIcon />
          </div>
          <div className={styles.statusHeader__title}>КТГ ID {ktgId}</div>
          {status !== 'idle' && (
            <div
              className={`${styles.statusHeader__status} ${
                styles[`statusHeader__status--${status}`]
              }`}
            >
              {status === 'recording' ? (
                <>
                  <div className={styles.statusHeader__indicator} />
                  <span>Идет запись</span>
                  <span>{currentTime}</span>
                </>
              ) : status === 'loading' ? (
                <>
                  <div className={styles.statusHeader__indicator} />
                  <span>Загрузка данных...</span>
                </>
              ) : (
                <>
                  <span>Проведено</span>
                  {isNewRecord && (
                    <span>
                      {new Date().toLocaleDateString('ru-RU', {
                        day: '2-digit',
                        month: '2-digit',
                        year: '2-digit',
                      })}
                    </span>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <div className={styles.statusHeader__actions}>
          {/* Скрытый input для выбора файла */}
          <input
            ref={fileInputRef}
            type='file'
            accept='.csv'
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />

          {status === 'idle' && isNewRecord && onUpload && (
            <Button
              variant='secondary'
              size='md'
              onClick={handleUploadClick}
              className={styles.statusHeader__uploadButton}
            >
              Загрузить
            </Button>
          )}

          {status === 'idle' && onStart && (
            <Button
              variant='primary'
              size='md'
              onClick={onStart}
              className={styles.statusHeader__completeButton}
            >
              Начать запись
            </Button>
          )}
          {status === 'recording' && onComplete && (
            <Button
              variant='primary'
              size='md'
              onClick={onComplete}
              className={styles.statusHeader__completeButton}
            >
              <div className={styles.statusHeader__icon} />
              Завершить
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};
