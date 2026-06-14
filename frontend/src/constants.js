export const VERDICT_COLORS = {
  success:            { bg: 'bg-green-100', text: 'text-green-800', label: 'Success' },
  success_unverified: { bg: 'bg-emerald-50', text: 'text-emerald-700', label: 'Success (Unverified)' },
  failure:            { bg: 'bg-red-100', text: 'text-red-800', label: 'Failure' },
  failure_corrupt:    { bg: 'bg-red-200', text: 'text-red-900', label: 'Failure (Corrupt)' },
  suspect:            { bg: 'bg-amber-100', text: 'text-amber-800', label: 'Suspect' },
  error:              { bg: 'bg-gray-100', text: 'text-gray-700', label: 'Error' },
};

export const STATUS_COLORS = {
  pending:   { bg: 'bg-gray-100', text: 'text-gray-600', label: 'Pending' },
  running:   { bg: 'bg-blue-100', text: 'text-blue-700', label: 'Running', animate: true },
  completed: { bg: 'bg-green-100', text: 'text-green-700', label: 'Completed' },
  failed:    { bg: 'bg-red-100', text: 'text-red-700', label: 'Failed' },
  skipped:   { bg: 'bg-yellow-50', text: 'text-yellow-700', label: 'Skipped' },
};

export const EVENT_LABELS = {
  suite_started:       'Suite Started',
  suite_running:       'Suite Running',
  scenarios_generated: 'Scenarios Generated',
  batch_started:       'Batch Started',
  batch_executed:      'Batch Executed',
  batch_scored:        'Batch Scored',
  patch_applied:       'Patch Applied',
  patch_failed:        'Patch Failed',
  batch_completed:     'Batch Completed',
  report_generating:   'Generating Report',
  suite_completed:     'Suite Completed',
  suite_failed:        'Suite Failed',
  parsing_yaml:        'Parsing YAML',
  yaml_validated:      'YAML Validated',
  initialized:         'Initialized',
};

export const DEFAULT_YAML = `agent_under_test:
  name: "CRM Booking Assistant"
  url: "wss://crm.example.com/chat"
  protocol: websocket
  description: >
    Books appointments for registered customers.
    Identifies customers by phone number.
    Only offers slots from the available calendar.
    Refuses unregistered callers politely.
  capabilities:
    - appointment booking
    - customer lookup
    - slot availability check
  known_limitations:
    - does not handle rescheduling
    - english and french only

world_state:
  entities:
    - id: customer_1
      phone: "+33612345678"
      name: "Isabelle Martin"
      verified: true
  catalog:
    available_slots:
      - "2026-05-20T10:00"
      - "2026-05-20T14:00"
  constraints:
    - "only verified entities can book"
    - "slots must be within 09:00-18:00"
    - "one booking per entity per day"
  context:
    current_time: "2026-05-16T08:00"
    language: "fr"
    channel: "voice"

test_config:
  total: 20
  positive: 14
  negative: 6

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514
`;
