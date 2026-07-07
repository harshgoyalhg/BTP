import { useState, useEffect } from 'react';
import { Shield, Activity, AlertTriangle, CheckCircle, Database } from 'lucide-react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';

const API_BASE = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState({ total_alerts: 0, high_severity: 0, distribution: {} });
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = async () => {
    try {
      const statsRes = await axios.get(`${API_BASE}/statistics`);
      setStats(statsRes.data);
      
      const alertsRes = await axios.get(`${API_BASE}/alerts?limit=10`);
      setAlerts(alertsRes.data);
    } catch (error) {
      console.error("Error fetching dashboard data. Is the backend running?", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  const distData = Object.keys(stats.distribution).map(key => ({
    name: key,
    count: stats.distribution[key]
  }));

  return (
    <div className="min-h-screen p-8">
      {/* Header */}
      <header className="flex items-center justify-between mb-10 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-cyber-dark rounded-xl neon-border">
            <Shield className="w-8 h-8 text-cyber-neon" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">AI SOC</h1>
            <p className="text-slate-400">Security Operations Center Dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-3 bg-cyber-dark px-4 py-2 rounded-lg border border-slate-800">
          <div className="w-3 h-3 rounded-full bg-cyber-safe animate-pulse"></div>
          <span className="text-sm font-medium text-cyber-safe">System Online</span>
        </div>
      </header>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-cyber-dark p-6 rounded-2xl border border-slate-800 flex items-center gap-6">
          <div className="p-4 bg-slate-800/50 rounded-full">
            <Activity className="w-8 h-8 text-cyber-neon" />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">Total Alerts Analyzed</p>
            <h2 className="text-4xl font-bold text-white mt-1">{stats.total_alerts.toLocaleString()}</h2>
          </div>
        </div>

        <div className="bg-cyber-dark p-6 rounded-2xl border border-slate-800 flex items-center gap-6">
          <div className="p-4 bg-red-900/20 rounded-full">
            <AlertTriangle className="w-8 h-8 text-cyber-alert" />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">High Severity Threats</p>
            <h2 className="text-4xl font-bold text-cyber-alert mt-1">{stats.high_severity.toLocaleString()}</h2>
          </div>
        </div>

        <div className="bg-cyber-dark p-6 rounded-2xl border border-slate-800 flex items-center gap-6">
          <div className="p-4 bg-emerald-900/20 rounded-full">
            <CheckCircle className="w-8 h-8 text-cyber-safe" />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">System Status</p>
            <h2 className="text-4xl font-bold text-cyber-safe mt-1">Defending</h2>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Alerts Feed */}
        <div className="lg:col-span-2 bg-cyber-dark rounded-2xl border border-slate-800 p-6">
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Database className="w-5 h-5 text-cyber-neon" /> 
            Recent Alerts Feed
          </h3>
          
          <div className="space-y-4">
            {alerts.length === 0 && !loading ? (
              <div className="text-center py-10 text-slate-500">No alerts detected yet. Send a test payload to see it here!</div>
            ) : (
              alerts.map(alert => (
                <div key={alert.id} className={`p-4 rounded-xl border ${alert.severity === 'HIGH' ? 'border-red-900/50 bg-red-950/20 alert-glow' : 'border-slate-800 bg-slate-900/50'} flex justify-between items-center transition-all hover:bg-slate-800`}>
                  <div>
                    <div className="flex items-center gap-3 mb-1">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${alert.severity === 'HIGH' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                        {alert.severity}
                      </span>
                      <h4 className="text-lg font-bold text-white">{alert.attack}</h4>
                    </div>
                    <p className="text-sm text-slate-400">{new Date(alert.time).toLocaleString()}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-slate-400 mb-1">AI Confidence</p>
                    <p className="text-xl font-bold text-cyber-neon">{(alert.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right Column: Charts & Analysis */}
        <div className="bg-cyber-dark rounded-2xl border border-slate-800 p-6 flex flex-col">
          <h3 className="text-xl font-bold text-white mb-6">Threat Distribution</h3>
          <div className="flex-1 min-h-[300px]">
            {distData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distData} layout="vertical" margin={{ top: 0, right: 0, left: 20, bottom: 0 }}>
                  <XAxis type="number" hide />
                  <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8' }} />
                  <RechartsTooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#020617', borderColor: '#334155', color: '#fff'}} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {distData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.name === 'BENIGN' ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500">Insufficient data</div>
            )}
          </div>
          
          <div className="mt-8 p-4 bg-slate-900 rounded-xl border border-slate-800">
            <h4 className="font-bold text-white mb-2">SHAP Explainability</h4>
            <p className="text-sm text-slate-400">
              When an alert fires, the AI uses SHAP values to explain the top 5 network features that led to the prediction. Check the individual alert details (WIP) to see the explanation!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
