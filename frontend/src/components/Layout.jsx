import { Link } from 'react-router-dom';

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="sticky top-0 z-20 bg-ink/95 backdrop-blur border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center gap-3 group">
              {/* Monogram tile — the one bit of brand identity */}
              <span className="grid place-items-center h-9 w-9 rounded-lg bg-brand-600 text-white font-bold text-sm tracking-tight shadow-card">
                ATA
              </span>
              <span className="flex flex-col leading-tight">
                <span className="text-sm font-semibold text-white">Agent Testing Agent</span>
                <span className="text-xs text-slate-400">Black-box E2E testing for conversational agents</span>
              </span>
            </Link>
            <Link to="/" className="btn-secondary border-white/15 bg-white/5 text-slate-200 hover:bg-white/10">
              New Run
            </Link>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">{children}</main>
    </div>
  );
}
