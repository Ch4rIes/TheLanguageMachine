import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Link, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import { ComparisonPage } from './pages/ComparisonPage';
import { ExperimentDetailPage } from './pages/ExperimentDetailPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { NewExperimentPage } from './pages/NewExperimentPage';
import { TokenizePage } from './pages/TokenizePage';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1 } },
});

function Nav() {
  return (
    <nav
      style={{
        background: '#1e293b',
        color: '#fff',
        padding: '12px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: 24,
      }}
    >
      <Link to="/" style={{ color: '#fff', textDecoration: 'none', fontWeight: 700, fontSize: 16 }}>
        CS336 Dashboard
      </Link>
      <Link to="/" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 14 }}>
        Experiments
      </Link>
      <Link to="/new" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 14 }}>
        New
      </Link>
      <Link to="/compare" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 14 }}>
        Compare
      </Link>
      <Link to="/tokenize" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 14 }}>
        Tokenize
      </Link>
    </nav>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Nav />
        <main style={{ maxWidth: 960, margin: '0 auto', padding: '24px 16px' }}>
          <Routes>
            <Route path="/" element={<ExperimentsPage />} />
            <Route path="/new" element={<NewExperimentPage />} />
            <Route path="/experiments/:id" element={<ExperimentDetailPage />} />
            <Route path="/compare" element={<ComparisonPage />} />
            <Route path="/tokenize" element={<TokenizePage />} />
          </Routes>
        </main>
      </Router>
    </QueryClientProvider>
  );
}
