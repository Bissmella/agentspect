import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import NewRun from './pages/NewRun';
import RunDashboard from './pages/RunDashboard';
import Report from './pages/Report';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<NewRun />} />
        <Route path="/runs/:id" element={<RunDashboard />} />
        <Route path="/runs/:id/report" element={<Report />} />
      </Routes>
    </Layout>
  );
}
