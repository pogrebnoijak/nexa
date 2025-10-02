import React from 'react';
import styles from './Button.module.scss';

export interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'icon';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  iconOnly?: boolean;
  iconLeft?: React.ReactNode;
  iconRight?: React.ReactNode;
  onClick?: () => void;
  className?: string;
  type?: 'button' | 'submit' | 'reset';
  style?: React.CSSProperties;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  iconOnly = false,
  iconLeft,
  iconRight,
  onClick,
  className = '',
  type = 'button',
  style,
}) => {
  const buttonClasses = [
    styles.button,
    styles[`button--${variant}`],
    styles[`button--${size}`],
    iconOnly && styles['button--icon-only'],
    disabled && styles['button--disabled'],
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button
      type={type}
      className={buttonClasses}
      disabled={disabled}
      onClick={onClick}
      style={style}
    >
      {iconLeft && <span className={styles.button__icon}>{iconLeft}</span>}
      {!iconOnly && <span className={styles.button__text}>{children}</span>}
      {iconRight && <span className={styles.button__icon}>{iconRight}</span>}
    </button>
  );
};
