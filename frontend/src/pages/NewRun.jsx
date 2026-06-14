import { useState } from 'react';
import YamlEditor from '../components/YamlEditor';
import ErrorAlert from '../components/ErrorAlert';
import { useCreateRun } from '../hooks/useRunQueries';
import { DEFAULT_YAML } from '../constants';

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
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900">New Test Run</h1>
        <button
          onClick={handleSubmit}
          disabled={createRun.isPending}
          className="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {createRun.isPending ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Submitting...
            </>
          ) : (
            'Run Tests'
          )}
        </button>
      </div>

      <ErrorAlert message={error} onDismiss={() => setError(null)} />

      <div className="mt-4">
        <YamlEditor value={yaml} onChange={setYaml} />
      </div>

      <p className="mt-3 text-xs text-gray-400">
        Paste your agent test configuration YAML above, then click Run Tests to start.
      </p>
    </div>
  );
}
