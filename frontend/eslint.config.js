// eslint.config.js
import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint'; // <— typescript-eslint для flat config
import { defineConfig, globalIgnores } from 'eslint/config';

export default defineConfig([
  // кого игнорим
  globalIgnores(['dist', 'node_modules']),

  // БАЗОВЫЕ ПРЕСЕТЫ — в корне массива, а не в "extends"
  js.configs.recommended,
  reactHooks.configs['recommended-latest'],
  reactRefresh.configs.vite,

  // JS/JSX
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: 'module',
      globals: globals.browser,
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
    },
  },

  // TS/TSX
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        // если нужны type-aware правила — укажи проект:
        // project: true,
        // tsconfigRootDir: import.meta.dirname,
      },
      ecmaVersion: 2023,
      sourceType: 'module',
      globals: globals.browser,
    },
    plugins: {
      '@typescript-eslint': tseslint.plugin,
    },
    // базовые TS-правила
    rules: {
      ...tseslint.configs.recommended.rules, // подключаем recommended прямо как rules
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^[A-Z_]' },
      ],
    },
  },
]);
