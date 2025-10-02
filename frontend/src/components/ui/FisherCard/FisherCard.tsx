import React from 'react';
import styles from './FisherCard.module.scss';

export interface FisherCardProps {
  score: number | null;
  maxScore?: number;
  status?: 'good' | 'suspicious' | 'pathological' | 'unknown';
  description?: string;
  className?: string;
}

export const FisherCard: React.FC<FisherCardProps> = ({
  score,
  maxScore = 10,
  status = 'good',
  description = 'Описание, почему мы так считаем и на что стоит обратить внимание',
  className = '',
}) => {
  const getStatusInfo = () => {
    switch (status) {
      case 'good':
        return {
          text: 'Хорошее',
          color: '#05A400',
        };
      case 'suspicious':
        return {
          text: 'Подозрительное',
          color: '#FF8D28',
        };
      case 'pathological':
        return {
          text: 'Патологическое',
          color: '#EF4640',
        };
      case 'unknown':
        return {
          text: 'Неизвестно',
          color: '#696F79',
        };
      default:
        return {
          text: 'Хорошее',
          color: '#05A400',
        };
    }
  };

  const statusInfo = getStatusInfo();

  return (
    <div className={`${styles.fisherCard} ${className}`}>
      <div className={styles.fisherCard__header}>
        <div className={styles.fisherCard__title}>Оценка по Фишеру</div>
        <div className={styles.fisherCard__status} style={{ backgroundColor: statusInfo.color }}>
          <div className={styles.fisherCard__statusText}>{statusInfo.text}</div>
        </div>
      </div>

      <div className={styles.fisherCard__content}>
        <div className={styles.fisherCard__score}>
          <div className={styles.fisherCard__scoreValue}>{score || 0}</div>
          <div className={styles.fisherCard__scoreMax}>/{maxScore}</div>
        </div>

        <div className={styles.fisherCard__separator}></div>

        <div className={styles.fisherCard__description}>{description}</div>
      </div>
    </div>
  );
};
