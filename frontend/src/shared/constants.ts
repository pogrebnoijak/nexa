// Тексты интерфейса
export const UI_TEXT = {
  // Заголовки
  FETAL_HR: 'ЧСС плода',
  UTERINE_TONE: 'Тонус матки',
  ANNOTATIONS: 'Аннотации',

  // Показатели
  MATERNAL_HR: 'ЧСС матери',
  BASAL_HR: 'Базальная ЧСС',
  VARIABILITY: 'Вариабельность',
  ACCELERATIONS: 'Акцелерации',
  DECELERATIONS: 'Децелерации',

  // Единицы измерения
  BPM: 'уд/мин',
  PIECES: 'шт',

  // Типы децелераций
  EARLY: 'Ранние:',
  LATE: 'Поздние:',
  VARIABLE: 'Вариаб.:',

  // События
  FETAL_MOVEMENT: 'Шевеление',

  // Навигация
  PAGE: 'Page',

  // Логотип
  MOSCOW_MEDICINE: 'Московская\nмедицина',
} as const;

// Цвета для графиков
export const CHART_COLORS = {
  ORANGE: '#FE822F',
  GREEN: '#3FEB41',
  RED: '#EF4640',
  YELLOW: '#FBC72B',
  BLUE: '#0087D6',
  BLACK: '#000000',
} as const;

// Данные для графиков
export const CHART_DATA = {
  // Y-ось для ЧСС плода (уд/мин)
  FETAL_HR_Y_AXIS: [220, 200, 180, 160, 140, 120, 100, 80, 60],

  // Y-ось для тонуса матки (%)
  UTERINE_TONE_Y_AXIS: [100, 80, 60, 40, 20, 0],

  // X-ось (время в минутах)
  TIME_AXIS: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],

  // Показатели
  INDICATORS: {
    FETAL_HR: 0,
    MATERNAL_HR: 0,
    BASAL_HR: 0,
    VARIABILITY: 0,
    ACCELERATIONS: 0,
    DECELERATIONS: 0,
    DECELERATIONS_BREAKDOWN: {
      EARLY: 0,
      LATE: 0,
      VARIABLE: 0,
    },
  },
} as const;

// Размеры компонентов
export const COMPONENT_SIZES = {
  HEADER_HEIGHT: 80,
  SIDEBAR_WIDTH: 511,
  CHART_HEIGHT: 278,
  UTERINE_CHART_HEIGHT: 183,
  CARD_HEIGHT: 128,
  ANNOTATIONS_HEIGHT: 396,
} as const;

// Анимации
export const ANIMATIONS = {
  DURATION_FAST: '0.15s',
  DURATION_NORMAL: '0.2s',
  DURATION_SLOW: '0.3s',
  EASING: 'ease-in-out',
} as const;

// Брейкпоинты
export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

// Z-index слои
export const Z_INDEX = {
  DROPDOWN: 1000,
  STICKY: 1020,
  FIXED: 1030,
  MODAL_BACKDROP: 1040,
  MODAL: 1050,
  POPOVER: 1060,
  TOOLTIP: 1070,
} as const;
