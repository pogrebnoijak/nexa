import React from 'react';
import styles from './IndicatorCard.module.scss';

export interface IndicatorCardProps {
  title: string;
  value: number | string;
  unit: string;
  icon?: React.ReactNode;
  breakdown?: {
    label: string;
    value: number;
    color?: string;
  }[];
  className?: string;
  onClick?: () => void;
}

export const IndicatorCard: React.FC<IndicatorCardProps> = ({
  title,
  value,
  unit,
  icon,
  breakdown,
  className = '',
  onClick,
}) => {
  const cardClasses = [styles.card, className, onClick ? styles.card__clickable : '']
    .filter(Boolean)
    .join(' ');

  return (
    <div className={cardClasses} onClick={onClick}>
      <div className={styles.card__info}>
        <h4 className={styles.card__title}>{title}</h4>
        <div className={styles.card__content}>
          <div className={styles.card__value}>
            <span className={styles.card__number}>{value}</span>
            <span className={styles.card__unit}>{unit}</span>
          </div>
          {breakdown && (
            <>
              <div className={styles.card__breakdownSeparator} />
              <div className={styles.card__breakdown}>
                <div className={styles.card__breakdownList}>
                  {breakdown.map((item, index) => (
                    <div key={index} className={styles.card__breakdownItem}>
                      <div className={styles.card__breakdownLabelContainer}>
                        {item.color && (
                          <div
                            className={styles.card__breakdownIndicator}
                            style={{ backgroundColor: item.color }}
                          />
                        )}
                        <span className={styles.card__breakdownLabel}>{item.label}</span>
                      </div>
                      <span className={styles.card__breakdownValue}>{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
