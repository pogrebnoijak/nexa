import React from 'react';
import { UI_TEXT } from '../../../shared/constants';
import styles from './AnnotationsPanel.module.scss';

export interface Annotation {
  time: string;
  event: string;
  value: string;
}

export interface AnnotationsPanelProps {
  annotations: Annotation[];
  className?: string;
}

export const AnnotationsPanel: React.FC<AnnotationsPanelProps> = ({
  annotations,
  className = '',
}) => {
  const panelClasses = [styles.panel, className].filter(Boolean).join(' ');

  return (
    <div className={panelClasses}>
      <h3 className={styles.panel__title}>{UI_TEXT.ANNOTATIONS}</h3>

      <div className={styles.panel__content}>
        {annotations.map((annotation, index) => (
          <div key={index} className={styles.panel__item}>
            <div className={styles.panel__time}>{annotation.time}</div>
            <div className={styles.panel__event}>{annotation.event}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
