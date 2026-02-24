import { useEffect, useRef, useState } from 'react';
import { metricsApi } from '../api';
import type { MetricPoint } from '../types';

export function useLiveMetrics(id: string, enabled: boolean) {
  const [points, setPoints] = useState<MetricPoint[]>([]);
  const esRef = useRef<EventSource | null>(null);

  // Initial snapshot
  useEffect(() => {
    metricsApi.snapshot(id).then((data) => setPoints(data)).catch(() => {});
  }, [id]);

  // SSE stream while running
  useEffect(() => {
    if (!enabled) {
      esRef.current?.close();
      esRef.current = null;
      return;
    }

    const lastIter = points.length > 0 ? Math.max(...points.map((p) => p.iteration)) : -1;
    const url = `http://localhost:8000/api/experiments/${id}/metrics/stream?last_iter=${lastIter}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const point: MetricPoint = JSON.parse(e.data);
        setPoints((prev) => {
          // Merge: deduplicate by (iteration, which fields are set)
          const key = (p: MetricPoint) =>
            `${p.iteration}-${p.train_loss !== undefined ? 'train' : ''}-${p.val_loss !== undefined ? 'val' : ''}`;
          const existing = new Set(prev.map(key));
          if (existing.has(key(point))) return prev;
          return [...prev, point].sort((a, b) => a.iteration - b.iteration);
        });
      } catch {}
    };

    es.onerror = () => {
      es.close();
    };

    return () => {
      es.close();
      esRef.current = null;
    };
    // We only want to re-open SSE when enabled changes, not on every point update
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, enabled]);

  return points;
}
