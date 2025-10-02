import { useEffect, useState } from 'react';
import { systemService } from '../services/systemService';

export const usePing = () => {
  const [isPinging, setIsPinging] = useState(false);
  const [pingResult, setPingResult] = useState<any>(null);
  const [pingError, setPingError] = useState<string | null>(null);

  const ping = async () => {
    setIsPinging(true);
    setPingError(null);

    try {
      const response = await systemService.ping();

      if (response.success) {
        setPingResult(response.data);
      } else {
        console.error('Ping failed:', response.error);
        setPingError(response.error || 'Ping request failed');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown ping error';
      console.error('Ping error:', errorMessage);
      setPingError(errorMessage);
    } finally {
      setIsPinging(false);
    }
  };

  // Автоматический ping при монтировании
  useEffect(() => {
    ping();
  }, []);

  return {
    ping,
    isPinging,
    pingResult,
    pingError,
  };
};
