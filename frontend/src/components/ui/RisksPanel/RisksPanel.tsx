import React from 'react';
import RadialChart from '../RadialChart';
import styles from './RisksPanel.module.scss';

export interface Risk {
  title: string;
  percentage: number;
  color: string;
  description: string;
}

export interface RisksPanelProps {
  risks: Risk[];
  className?: string;
}

export const RisksPanel: React.FC<RisksPanelProps> = ({ risks, className = '' }) => {
  const panelClasses = [styles.panel, className].filter(Boolean).join(' ');

  return (
    <div className={panelClasses}>
      <h3 className={styles.panel__title}>Риски</h3>

      <div className={styles.panel__content}>
        {risks.map((risk, index) => (
          <div key={index} className={styles.risk}>
            <h4 className={styles.risk__title}>{risk.title}</h4>
            <div className={styles.risk__content}>
              <div className={styles.risk__chart}>
                <RadialChart
                  value={risk.percentage}
                  maxValue={100}
                  size={88}
                  strokeWidth={8}
                  color={risk.color}
                  backgroundColor='#f4f4f5'
                >
                  <div className={styles.risk__percentage}>{risk.percentage}%</div>
                </RadialChart>
              </div>
              <p className={styles.risk__description}>{risk.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
