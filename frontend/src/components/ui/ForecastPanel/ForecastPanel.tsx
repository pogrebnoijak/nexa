import React from 'react';
import styles from './ForecastPanel.module.scss';

export interface Forecast {
  title: string;
  description: string;
}

export interface ForecastPanelProps {
  forecasts: Forecast[];
  className?: string;
}

export const ForecastPanel: React.FC<ForecastPanelProps> = ({ forecasts, className = '' }) => {
  const panelClasses = [styles.panel, className].filter(Boolean).join(' ');

  return (
    <div className={panelClasses}>
      <h3 className={styles.panel__title}>Прогноз</h3>

      <div className={styles.panel__content}>
        {forecasts.map((forecast, index) => (
          <div key={index} className={styles.forecast}>
            <h4 className={styles.forecast__title}>{forecast.title}</h4>
            <p className={styles.forecast__description}>{forecast.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
