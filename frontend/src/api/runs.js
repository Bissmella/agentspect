import client from './client';

export const createRun = (yamlContent) =>
  client.post('/api/runs', { yaml_content: yamlContent });

export const getRun = (id) =>
  client.get(`/api/runs/${id}`);

export const getReport = (id) =>
  client.get(`/api/runs/${id}/report`);

export const getScenarios = (id) =>
  client.get(`/api/runs/${id}/scenarios`);

export const getTranscript = (runId, scenarioId) =>
  client.get(`/api/runs/${runId}/scenarios/${scenarioId}/transcript`);
