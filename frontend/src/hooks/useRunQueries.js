import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { createRun, getRun, getReport, getScenarios, getTranscript } from '../api/runs';

export function useRun(id, { poll = false } = {}) {
  return useQuery({
    queryKey: ['run', id],
    queryFn: () => getRun(id),
    enabled: !!id,
    refetchInterval: poll ? 5000 : false,
  });
}

export function useReport(id) {
  return useQuery({
    queryKey: ['report', id],
    queryFn: () => getReport(id),
    enabled: !!id,
  });
}

export function useScenarios(id) {
  return useQuery({
    queryKey: ['scenarios', id],
    queryFn: () => getScenarios(id),
    enabled: !!id,
  });
}

export function useTranscript(runId, scenarioId) {
  return useQuery({
    queryKey: ['transcript', runId, scenarioId],
    queryFn: () => getTranscript(runId, scenarioId),
    enabled: !!runId && !!scenarioId,
  });
}

export function useCreateRun() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (yamlContent) => createRun(yamlContent),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['run'] });
      navigate(`/runs/${data.suite_id}`);
    },
  });
}
