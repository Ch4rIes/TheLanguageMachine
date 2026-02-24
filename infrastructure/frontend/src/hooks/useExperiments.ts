import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { experimentsApi } from '../api';
import type { ExperimentCreate } from '../types';

export function useExperiments() {
  return useQuery({
    queryKey: ['experiments'],
    queryFn: experimentsApi.list,
    refetchInterval: 5000,
  });
}

export function useExperiment(id: string) {
  return useQuery({
    queryKey: ['experiments', id],
    queryFn: () => experimentsApi.get(id),
    refetchInterval: 5000,
  });
}

export function useExperimentStatus(id: string, enabled: boolean) {
  return useQuery({
    queryKey: ['experiments', id, 'status'],
    queryFn: () => experimentsApi.status(id),
    refetchInterval: enabled ? 5000 : false,
    enabled,
  });
}

export function useCreateExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ExperimentCreate) => experimentsApi.create(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['experiments'] }),
  });
}

export function useLaunchExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => experimentsApi.launch(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['experiments'] }),
  });
}

export function useStopExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => experimentsApi.stop(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['experiments'] }),
  });
}

export function useDeleteExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => experimentsApi.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['experiments'] }),
  });
}

export function useCheckpoints(id: string) {
  return useQuery({
    queryKey: ['experiments', id, 'checkpoints'],
    queryFn: () => experimentsApi.checkpoints(id),
  });
}

export function useLog(id: string, enabled: boolean) {
  return useQuery({
    queryKey: ['experiments', id, 'log'],
    queryFn: () => experimentsApi.log(id),
    refetchInterval: enabled ? 3000 : false,
    enabled,
  });
}
