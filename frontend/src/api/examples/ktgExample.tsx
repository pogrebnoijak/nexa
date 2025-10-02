import React from 'react';
import { useKTGList, useKTGDetail, useKTGActions, useWebSocket, useKTGDataStream } from '../index';

// Пример использования API в компоненте
export const KTGExample: React.FC = () => {
  // Получение списка КТГ записей
  const { data: ktgList, loading: listLoading, error: listError, refetch } = useKTGList();

  // Получение деталей КТГ записи
  const { data: ktgDetail, loading: detailLoading, error: detailError } = useKTGDetail('123455');

  // Действия с КТГ
  const { startRecording, stopRecording, createKTG, loading: actionLoading } = useKTGActions();

  // WebSocket подключение
  const { isConnected, error: wsError } = useWebSocket();

  // Поток данных КТГ
  const { data: ktgData, isStreaming } = useKTGDataStream('123455');

  const handleStartRecording = async () => {
    const success = await startRecording('123455');
    if (success) {
    }
  };

  const handleStopRecording = async () => {
    const success = await stopRecording('123455');
    if (success) {
    }
  };

  const handleCreateKTG = async () => {
    const newKTG = await createKTG({
      patientId: 'patient123',
      patientName: 'Test Patient',
      weeks: '32 недели',
      status: 'recording',
      startTime: new Date().toISOString(),
    });

    if (newKTG) {
    }
  };

  return (
    <div>
      <h2>KTG API Example</h2>

      {/* WebSocket статус */}
      <div>
        <p>WebSocket: {isConnected ? 'Connected' : 'Disconnected'}</p>
        {wsError && <p>WebSocket Error: {wsError}</p>}
      </div>

      {/* Список КТГ записей */}
      <div>
        <h3>KTG Records</h3>
        {listLoading && <p>Loading...</p>}
        {listError && <p>Error: {listError}</p>}
        {ktgList && (
          <ul>
            {ktgList.map((record) => (
              <li key={record.id}>
                {record.patientName} - {record.status}
              </li>
            ))}
          </ul>
        )}
        <button onClick={refetch}>Refresh</button>
      </div>

      {/* Детали КТГ */}
      <div>
        <h3>KTG Detail</h3>
        {detailLoading && <p>Loading...</p>}
        {detailError && <p>Error: {detailError}</p>}
        {ktgDetail && (
          <div>
            <p>Patient: {ktgDetail.patientName}</p>
            <p>Status: {ktgDetail.status}</p>
            <p>Weeks: {ktgDetail.weeks}</p>
          </div>
        )}
      </div>

      {/* Действия */}
      <div>
        <h3>Actions</h3>
        <button onClick={handleStartRecording} disabled={actionLoading}>
          Start Recording
        </button>
        <button onClick={handleStopRecording} disabled={actionLoading}>
          Stop Recording
        </button>
        <button onClick={handleCreateKTG} disabled={actionLoading}>
          Create KTG
        </button>
      </div>

      {/* Поток данных */}
      <div>
        <h3>KTG Data Stream</h3>
        <p>Streaming: {isStreaming ? 'Yes' : 'No'}</p>
        <p>Data points: {ktgData.length}</p>
        {ktgData.length > 0 && (
          <div>
            <p>Latest data:</p>
            <pre>{JSON.stringify(ktgData[ktgData.length - 1], null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
};
