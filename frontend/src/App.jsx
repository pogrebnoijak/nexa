import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { Toaster } from 'sonner';
import { MainPage } from './pages/MainPage';
import { HomePage } from './pages/HomePage';
import SidebarMenu from './components/ui/SidebarMenu';
import SidebarHandle from './components/ui/SidebarHandle';
import './styles/index.scss';

function AppContent() {
  const location = useLocation();
  const sidebarRef = useRef(null);
  const handleRef = useRef(null);

  const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
    // Восстанавливаем состояние из localStorage
    const saved = localStorage.getItem('sidebar-open');
    if (saved) {
      return JSON.parse(saved);
    }
    // По умолчанию открыто на главной странице
    return location.pathname === '/';
  });

  // Ping запрос при открытии приложения

  // Единая переменная для определения состояния меню
  const isMainPage = location.pathname === '/';
  const isMenuOpen = isSidebarOpen;

  // Сохраняем состояние меню в localStorage
  useEffect(() => {
    localStorage.setItem('sidebar-open', JSON.stringify(isSidebarOpen));
  }, [isSidebarOpen]);

  // Автоматически управляем меню в зависимости от страницы
  useEffect(() => {
    if (isMainPage) {
      // На главной странице меню открыто
      setIsSidebarOpen(true);
    } else {
      // На страницах КТГ меню закрыто
      setIsSidebarOpen(false);
    }
  }, [isMainPage]);

  // Показываем ручку только на детальных страницах КТГ
  const showHandle = location.pathname.startsWith('/ktg/');

  // Обработчик клика вне меню для страниц КТГ
  useEffect(() => {
    const handleClickOutside = (event) => {
      // Работаем только на страницах КТГ и если меню открыто
      if (!showHandle || !isSidebarOpen) return;

      // Проверяем, что клик не по меню и не по ручке
      const isClickOnSidebar = sidebarRef.current && sidebarRef.current.contains(event.target);
      const isClickOnHandle = handleRef.current && handleRef.current.contains(event.target);

      if (!isClickOnSidebar && !isClickOnHandle) {
        setIsSidebarOpen(false);
      }
    };

    // Добавляем обработчик только на страницах КТГ
    if (showHandle) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [showHandle, isSidebarOpen]);

  return (
    <>
      <SidebarMenu ref={sidebarRef} isOpen={isMenuOpen} />
      <SidebarHandle
        ref={handleRef}
        isOpen={isMenuOpen}
        onToggle={setIsSidebarOpen}
        isVisible={showHandle}
      />
      <Routes>
        <Route path='/' element={<MainPage />} />
        <Route path='/ktg/:id' element={<HomePage />} />
      </Routes>
      <Toaster position='top-right' />
    </>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
