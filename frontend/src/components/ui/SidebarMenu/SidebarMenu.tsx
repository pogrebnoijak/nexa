import React, { forwardRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { NexaIcon, SunIcon, BookIcon, FolderIcon } from '../../../icons';
import styles from './SidebarMenu.module.scss';

interface SidebarMenuProps {
  isOpen: boolean;
}

const SidebarMenu = forwardRef<HTMLDivElement, SidebarMenuProps>(({ isOpen }, ref) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <div ref={ref} className={`${styles.menu} ${isOpen ? styles.menu__open : ''}`}>
      <div className={styles.menu__content}>
        <div className={styles.menu__header}>
          <div className={styles.menu__logo}>
            <NexaIcon />
          </div>
        </div>

        <div className={styles.menu__section}>
          <div className={styles.menu__items}>
            <div
              className={`${styles.menu__item} ${isActive('/') ? styles.menu__itemActive : ''}`}
              onClick={() => handleNavigation('/')}
            >
              <BookIcon className={styles.menu__itemIcon} />
              <span className={styles.menu__itemText}>Главная</span>
            </div>
            <div className={`${styles.menu__item} ${styles.menu__itemDisabled}`}>
              <FolderIcon className={styles.menu__itemIcon} />
              <span className={styles.menu__itemText}>Записи на прием</span>
            </div>
            <div className={`${styles.menu__item} ${styles.menu__itemDisabled}`}>
              <BookIcon className={styles.menu__itemIcon} />
              <span className={styles.menu__itemText}>Пациенты</span>
            </div>
            <div className={`${styles.menu__item} ${styles.menu__itemDisabled}`}>
              <FolderIcon className={styles.menu__itemIcon} />
              <span className={styles.menu__itemText}>Настройки</span>
            </div>
            <div className={`${styles.menu__item} ${styles.menu__itemDisabled}`}>
              <FolderIcon className={styles.menu__itemIcon} />
              <span className={styles.menu__itemText}>Помощь</span>
            </div>
          </div>
        </div>

        <div className={styles.menu__footer}>
          <div className={styles.menu__item}>
            <FolderIcon className={styles.menu__itemIcon} />
            <span className={styles.menu__itemText}>Настройки аппарата</span>
          </div>
        </div>
      </div>
    </div>
  );
});

SidebarMenu.displayName = 'SidebarMenu';

export default SidebarMenu;
