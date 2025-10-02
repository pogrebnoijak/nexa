import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Header } from '../../components/layout/Header';
import { UserIcon, CalendarIcon } from '../../icons';
import { analysisService } from '../../api';
import styles from './MainPage.module.scss';

export const MainPage: React.FC = () => {
  // Состояние для данных с API
  const [ktgIds, setKtgIds] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Загрузка данных при монтировании компонента
  useEffect(() => {
    const loadKtgIds = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const response = await analysisService.getAvailableIds();

        if (response.success && response.data) {
          // Проверяем, что data является массивом
          let data: string[] = [];
          const responseData = response.data as any;

          if (Array.isArray(responseData)) {
            data = responseData;
          } else if (
            responseData &&
            typeof responseData === 'object' &&
            Array.isArray(responseData.data)
          ) {
            data = responseData.data;
          }

          setKtgIds(data);
        } else {
          setError(response.error || 'Не удалось загрузить список КТГ');
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Ошибка при загрузке данных';
        setError(errorMessage);
      } finally {
        setIsLoading(false);
      }
    };

    loadKtgIds();
  }, []);

  // Генерируем записи КТГ на основе полученных ID
  const ktgRecords = (Array.isArray(ktgIds) ? ktgIds : []).map((id, index) => ({
    id,
    patientName: `Пациент ${index + 1}`, // Временное имя, пока нет данных о пациентах
    weeks: `${28 + (index % 8)} недель`, // Временные данные
    status: 'Неизвестно', // Серый статус для всех записей
    date:
      new Date(Date.now() - index * 24 * 60 * 60 * 1000).toLocaleDateString('ru-RU', {
        day: '2-digit',
        month: '2-digit',
        year: '2-digit',
      }) +
      ', ' +
      new Date(Date.now() - index * 60 * 60 * 1000).toLocaleTimeString('ru-RU', {
        hour: '2-digit',
        minute: '2-digit',
      }) +
      '-' +
      new Date(Date.now() - index * 60 * 60 * 1000 + 50 * 60 * 1000).toLocaleTimeString('ru-RU', {
        hour: '2-digit',
        minute: '2-digit',
      }),
  }));

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Хорошее':
        return '#05A400';
      case 'Удовлетв.':
        return '#FF8D28';
      case 'Плохое':
        return '#FF5F57';
      case 'Неизвестно':
        return '#696F79'; // Серый цвет для неизвестного статуса
      default:
        return '#696F79'; // Серый цвет по умолчанию
    }
  };

  // Генерация нового 8-символьного ID
  const generateNewKTGId = (): string => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let result = '';
    for (let i = 0; i < 8; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  };

  return (
    <div className={styles.page}>
      <Header breadcrumbItems={['Кардиотокография']} hideLogo={true} />

      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.titleSection}>
            <div className={styles.title}>
              <h1>Кардиотокография</h1>
            </div>
            <Link to='/ktg/new' className={styles.newKTGButton}>
              Новое КТГ
            </Link>
          </div>

          <div className={styles.recordsList}>
            {isLoading && <div className={styles.loadingMessage}>Загрузка списка КТГ...</div>}

            {error && <div className={styles.errorMessage}>{error}</div>}

            {!isLoading && !error && ktgRecords.length === 0 && (
              <div className={styles.emptyMessage}>Нет доступных записей КТГ</div>
            )}

            {!isLoading &&
              !error &&
              ktgRecords.map((record) => (
                <Link key={record.id} to={`/ktg/${record.id}`} className={styles.recordCard}>
                  <div className={styles.recordId}>КТГ ID {record.id}</div>
                  <div className={styles.patientInfo}>
                    <div className={styles.patientIcon}>
                      <UserIcon />
                    </div>
                    <div className={styles.patientName}>{record.patientName}</div>
                  </div>
                  <div className={styles.weeks}>{record.weeks}</div>
                  <div
                    className={styles.status}
                    style={{ backgroundColor: getStatusColor(record.status) }}
                  >
                    {record.status}
                  </div>
                  <div className={styles.dateInfo}>
                    <div className={styles.dateIcon}>
                      <CalendarIcon />
                    </div>
                    <div className={styles.date}>{record.date}</div>
                  </div>
                </Link>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
};
