import React, { useEffect, useState } from 'react';
import './styles.css';
import { BarChart, ScatterChart, Heatmap, sortRows, byStrategy, label } from './Charts';

const API = 'http://localhost:8000/api';
const fmtNgn = new Intl.NumberFormat('en-NG', { style: 'currency', currency: 'NGN', maximumFractionDigits: 0 });
const fmtPct = (v: number | null | undefined) => v != null ? `${v > 0 ? '+' : ''}${Number(v).toFixed(1)}%` : '-';

export default function App() {
    const [meta, setMeta] = useState<any>({ patterns: [], scenarios: [], pv_kws: [], battery_kwhs: [], strategies: [] });
    const [filter, setFilter] = useState({ pattern: 'uniform', scenario: 'severe', pv: 3.0, battery: 2.5, strat: 'oracle' });
    const [summary, setSummary] = useState([]);
    const [split, setSplit] = useState([]);
    const [grid, setGrid] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
      fetch(`${API}/meta`).then(r => r.json()).then(m => setMeta({
          ...m, scenarios: ['mild', 'moderate', 'severe', 'extreme'].filter(s => m.scenarios.includes(s)),
          strategies: [...m.strategies].sort(byStrategy),
      })).catch(() => setError('Could not reach the server on port 8000.'));
  }, []);

  useEffect(() => {
      const ctrl = new AbortController();
      setLoading(true); setError('');
      const { pattern, scenario, pv, battery, strat } = filter;
      const q = `pattern=${pattern}&scenario=${scenario}&pv_kw=${pv}&battery_kwh=${battery}`;
      const get = (p: string) => fetch(`${API}/${p}`, { signal: ctrl.signal }).then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); });
      Promise.all([get(`summary?source=sweep&${q}`), get(`split?${q}`), get(`grid?pattern=${pattern}&scenario=${scenario}&strategy=${strat}`)])
          .then(([sum, spl, grd]) => { setSummary(sortRows(sum)); setSplit(sortRows(spl)); setGrid(grd); setLoading(false); })
          .catch(e => { if (e.name !== 'AbortError') { setError('Could not reach the server on port 8000.'); setLoading(false); } });
      return () => ctrl.abort();
  }, [filter]);

  const row = (s: string) => summary.find((d: any) => d.strategy === s);
  const reactive = row('reactive'), w50 = row('helios_w50'), oracle = row('oracle');
  const bestTest = Math.max(...split.filter((s: any) => s.strategy !== 'reactive').map((s: any) => s.test_mean ?? -Infinity));


    return (
        <div className="app-container">
          <header className="header">
              <div><h1>HELIOS</h1><p>Battery dispatch strategies · 30 simulated outage years</p></div>
              <div className="filters">
                  <label>Outage pattern<select value={filter.pattern} onChange={e => setFilter({ ...filter, pattern: e.target.value })}>
                      {meta.patterns.map((o: string) => <option key={o} value={o}>{o}</option>)}</select></label>
                  <label>Grid scenario<select value={filter.scenario} onChange={e => setFilter({ ...filter, scenario: e.target.value })}>
                      {meta.scenarios.map((o: string) => <option key={o} value={o}>{o}</option>)}</select></label>
                  <label>Solar panels<select value={filter.pv} onChange={e => setFilter({ ...filter, pv: Number(e.target.value) })}>
                      {meta.pv_kws.map((o: number) => <option key={o} value={o}>{o} kW</option>)}</select></label>
                  <label>Battery<select value={filter.battery} onChange={e => setFilter({ ...filter, battery: Number(e.target.value) })}>
                      {meta.battery_kwhs.map((o: number) => <option key={o} value={o}>{o} kWh</option>)}</select></label>
              </div>
          </header>

          {error && <div className="err">{error}</div>}

          <div className="kpis">
              <div className="kpi"><span>Reactive baseline</span><b>{reactive ? fmtNgn.format(reactive.cost_ngn) : '-'}</b><small>cost per year</small></div>
              <div className="kpi accent"><span>w50 saving</span><b>{fmtPct(w50?.saved_pct_mean)}</b><small>vs reactive</small></div>
              <div className="kpi"><span>Generator hours avoided (w50)</span><b>{w50 ? Number(w50.gen_avoided).toFixed(0) : '-'}</b><small>of {reactive ? Number(reactive.gen_hours).toFixed(0) : '-'} h a year</small></div>
              <div className="kpi violet"><span>Oracle ceiling</span><b>{fmtPct(oracle?.saved_pct_mean)}</b><small>perfect foresight</small></div>
          </div>

            <div className={grid${loading ? ' dim' : ''}}>
                <div className="card">
                    <h2>Mean Savings vs Reactive</h2>
                    <BarChart data={summary} />
                </div>
                <div className="card">
                    <h2>Total Cost vs Gen Hours</h2>
                    <ScatterChart data={summary} />
                </div>
                <div className="card">
                    <h2>
                        Heatmap: Savings Grid
                        <select style={{ marginLeft: '1rem' }} value={filter.strat} onChange={e => setFilter({...filter, strat: e.target.value})}>
                            {meta.strategies.map((s: string) => <option key={s}>{s}</option>)}
                        </select>
                    </h2>
                    <Heatmap data={grid} />
                </div>

                <div className="card" style={{ gridColumn: '1 / -1' }}>
                    <h2>Tune vs Test Splits</h2>
                    <table>
                        <thead>
                            <tr><th>Strategy</th><th>Tune Mean</th><th>Tune Std</th><th>Test Mean</th><th>Test Std</th></tr>
                        </thead>
                        <tbody>
                            {split.map((s: any) => (
                                <tr key={s.strategy}>
                                    <td>{s.strategy}</td>
                                    <td>{fmtPct(s.tune_mean)}</td><td>{fmtPct(s.tune_std)}</td>
                                    <td>{fmtPct(s.test_mean)}</td><td>{fmtPct(s.test_std)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                <div className="card" style={{ gridColumn: '1 / -1' }}>
                    <h2>Scenario Summary Table</h2>
                    <table>
                        <thead>
                           <tr><th>Strategy</th><th>Total Cost (NGN)</th><th>Gen Hours</th><th>Saved %</th><th>Seeds &gt; Base</th></tr>
                        </thead>
                        <tbody>
                            {summary.map((s: any) => (
                                <tr key={s.strategy}>
                                    <td>{s.strategy}</td>
                                    <td>{fmtNgn.format(s.cost_ngn)}</td>
                                    <td>{Number(s.gen_hours).toFixed(1)}</td>
                                    <td>{fmtPct(s.saved_pct_mean)}</td>
                                    <td>{s.seeds_beating_baseline}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
