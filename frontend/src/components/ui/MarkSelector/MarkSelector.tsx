import React, { useState, useEffect } from 'react';
import styles from './MarkSelector.module.scss';

interface MarkOption {
  value: string;
  label: string;
}

interface MarkSelectorProps {
  onMarkSelect: (mark: string, timestamp: number) => void;
  lastTimestamp?: number;
  className?: string;
}

const DEFAULT_OPTIONS: MarkOption[] = [{ value: 'movement', label: 'Шевеление' }];

export const MarkSelector: React.FC<MarkSelectorProps> = ({
  onMarkSelect,
  lastTimestamp = 0,
  className = '',
}) => {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState('');
  const [options, setOptions] = useState<MarkOption[]>(DEFAULT_OPTIONS);
  const [searchTerm, setSearchTerm] = useState('');

  // Загружаем опции из localStorage при монтировании
  useEffect(() => {
    const savedOptions = localStorage.getItem('mark-selector-options');
    if (savedOptions) {
      try {
        const parsed = JSON.parse(savedOptions);
        setOptions([
          ...DEFAULT_OPTIONS,
          ...parsed.filter(
            (opt: MarkOption) => !DEFAULT_OPTIONS.some((def) => def.value === opt.value)
          ),
        ]);
      } catch (error) {
        // Ошибка загрузки опций из localStorage
      }
    }
  }, []);

  // Сохраняем опции в localStorage при изменении
  const saveOptionsToStorage = (newOptions: MarkOption[]) => {
    const customOptions = newOptions.filter(
      (opt) => !DEFAULT_OPTIONS.some((def) => def.value === opt.value)
    );
    localStorage.setItem('mark-selector-options', JSON.stringify(customOptions));
  };

  // Фильтруем опции по поисковому запросу
  const filteredOptions = options.filter((option) =>
    option.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Обработка выбора опции
  const handleSelect = (selectedValue: string) => {
    setOpen(false);
    setSearchTerm('');

    const selectedOption = options.find((opt) => opt.value === selectedValue);
    if (selectedOption) {
      onMarkSelect(selectedOption.label, lastTimestamp);
    }
  };

  // Обработка нажатия Enter для создания новой опции
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && searchTerm.trim() && filteredOptions.length === 0) {
      e.preventDefault();
      const newValue = searchTerm.trim().toLowerCase().replace(/\s+/g, '-');
      const newOption: MarkOption = {
        value: newValue,
        label: searchTerm.trim(),
      };

      const newOptions = [...options, newOption];
      setOptions(newOptions);
      saveOptionsToStorage(newOptions);

      setOpen(false);
      setSearchTerm('');
      onMarkSelect(newOption.label, lastTimestamp);
    }
  };

  return (
    <div className={`${styles.selector} ${className}`}>
      <button className={styles.trigger} onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={styles.triggerText}>Поставить метку</span>
      </button>

      {open && (
        <div className={styles.content}>
          <div className={styles.search}>
            <input
              type='text'
              placeholder='Поиск меток...'
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyDown={handleKeyDown}
              className={styles.searchInput}
              autoFocus
            />
          </div>

          <div className={styles.list}>
            {filteredOptions.length > 0 ? (
              filteredOptions.map((option) => (
                <button
                  key={option.value}
                  className={styles.item}
                  onClick={() => handleSelect(option.value)}
                >
                  <span>{option.label}</span>
                </button>
              ))
            ) : searchTerm.trim() ? (
              <div className={styles.empty}>
                <span>Метка не найдена.</span>
                <span className={styles.emptyHint}>
                  Нажмите Enter, чтобы создать "{searchTerm.trim()}"
                </span>
              </div>
            ) : (
              <div className={styles.empty}>
                <span>Нет доступных меток.</span>
              </div>
            )}
          </div>
        </div>
      )}

      {open && <div className={styles.overlay} onClick={() => setOpen(false)} />}
    </div>
  );
};
