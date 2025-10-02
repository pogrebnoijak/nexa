import React from 'react';
import styles from './RadialChart.module.scss';

interface RadialChartProps {
  value: number;
  maxValue?: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  backgroundColor?: string;
  children?: React.ReactNode;
  className?: string;
}

const RadialChart: React.FC<RadialChartProps> = ({
  value,
  maxValue = 100,
  size = 88,
  strokeWidth = 8,
  color = '#42c1c7',
  backgroundColor = '#f4f4f5',
  children,
  className = '',
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(value / maxValue, 1);
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference * (1 - progress);

  return (
    <div className={`${styles.container} ${className}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} className={styles.svg}>
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill='none'
          stroke={backgroundColor}
          strokeWidth={strokeWidth}
          className={styles.background}
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill='none'
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap='round'
          className={styles.progress}
          style={{
            transform: 'rotate(-90deg)',
            transformOrigin: `${size / 2}px ${size / 2}px`,
          }}
        />
      </svg>
      {children && <div className={styles.content}>{children}</div>}
    </div>
  );
};

export default RadialChart;
