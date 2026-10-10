import { useState } from 'react';
import YamlEditor from '../components/YamlEditor';
import ErrorAlert from '../components/ErrorAlert';
import Spinner from '../components/Spinner';
import { useCreateRun } from '../hooks/useRunQueries';
import { DEFAULT_YAML } from '../constants';

const STEPS = [
  'Generate positive & negative scenarios from your world_state',
  'Drive real conversations against the agent over HTTP / WebSocket',
  'Score each run and probe for state corruption',
  'Synthesise verdicts, metrics and a failure analysis',
];

export default function NewRun() {
  const [yaml, setYaml] = useState(DEFAULT_YAML);
  const [error, setError] = useState(null);
  const createRun = useCreateRun();

  const handleSubmit = () => {
    setError(null);
    createRun.mutate(yaml, {
      onError: (err) => setError(err.detail || 'An unexpected error occurred.'),
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">New test run</h1>
        <p className="mt-1 text-sm text-slate-500">
          Describe the agent and its world, then run a suite against it — no access to its internals required.
        </p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* Editor */}
        <div className="lg:col-span-2 card overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5">
            <span className="panel-title">Configuration · YAML</span>
            <div className="flex items-center gap-2">
              <button className="btn-secondary px-2.5 py-1 text-xs" onClick={() => setYaml(DEFAULT_YAML)}>
                Load example
              </button>
              <button
                className="btn-secondary px-2.5 py-1 text-xs"
                onClick={() => setYaml('')}
                disabled={!yaml}
              >
                Clear
              </button>
            </div>
          </div>
          <YamlEditor value={yaml} onChange={setYaml} />
        </div>

        {/* Side panel */}
        <aside className="card p-5 space-y-5">
          <div>
            <h2 className="panel-title mb-3">What happens on a run</h2>
            <ol className="space-y-3">
              {STEPS.map((step, i) => (
                <li key={i} className="flex gap-3 text-sm text-slate-600">
                  <span className="grid place-items-center h-5 w-5 shrink-0 rounded-full bg-brand-50 text-brand-700 text-xs font-semibold">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </div>

          <div className="border-t border-slate-200 pt-4">
            <button className="btn-primary w-full" onClick={handleSubmit} disabled={createRun.isPending || !yaml.trim()}>
              {createRun.isPending ? (
                <>
                  <Spinner className="h-4 w-4 text-white" /> Submitting…
                </>
              ) : (
                'Run tests'
              )}
            </button>
            <p className="mt-2 text-xs text-slate-400">
              Provider and model come from <span className="font-mono text-slate-500">llm_config</span> in the YAML.
              API keys are read from the server environment.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
