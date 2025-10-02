import React from 'react';
import { Button } from '../../ui/Button';
import { NexaIcon, ChevronDownIcon, ChevronRightIcon, SunIcon, ProfileIcon } from '../../../icons';
import styles from './Header.module.scss';

export interface HeaderProps {
  className?: string;
  breadcrumbItems?: string[];
  hideLogo?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ breadcrumbItems, hideLogo = false }) => {
  return (
    <header>
      <div className={styles.header__content}>
        {/* Логотип и навигация */}
        <div className={styles.header__left}>
          {!hideLogo && (
            <div className={styles.header__logo}>
              <NexaIcon />
            </div>
          )}

          <nav className={styles.header__nav}>
            <Breadcrumb items={breadcrumbItems} />
          </nav>
        </div>

        {/* Правые элементы */}
        <div className={styles.header__right}>
          <div className={styles.header__controls}>
            <ThemeToggle />
          </div>
          <div className={styles.header__avatarBlock}>
            <div className={styles.header__avatar}>
              <ProfileIcon />
            </div>
            <ChevronDownIcon />
          </div>
        </div>
      </div>
    </header>
  );
};

// Компонент хлебных крошек
const Breadcrumb: React.FC<{ items?: string[] }> = ({ items }) => {
  const breadcrumbItems = items || [
    'Пациенты',
    'Антонова Светлана Игоревна, 32 неделя',
    'КТГ ID 123455',
  ];

  return (
    <div className={styles.breadcrumb}>
      {breadcrumbItems.map((item, index) => (
        <React.Fragment key={index}>
          <span
            className={`${styles.breadcrumb__item} ${
              index === breadcrumbItems.length - 1 ? styles['breadcrumb__item--active'] : ''
            }`}
          >
            {item}
          </span>
          {index < breadcrumbItems.length - 1 && <ChevronRightIcon />}
        </React.Fragment>
      ))}
    </div>
  );
};

// Компонент переключателя темы
const ThemeToggle: React.FC = () => (
  <div className={styles.themeToggle}>
    <div className={styles.themeToggle__track}>
      <div className={styles.themeToggle__thumb}>
        <SunIcon />
      </div>
    </div>
  </div>
);
