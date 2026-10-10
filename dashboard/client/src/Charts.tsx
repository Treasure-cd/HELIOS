import React from 'react';

export interface Row { strategy: string; cost_ngn?: number; gen_hours?: number; saved_pct_mean?: number | null; saved_pct_std?: number | null; [k: string]: any }
export interface Cell { pv_kw: number; battery_kwh: number; value: number | null }

const ORDER = ['naive', 'reactive', 'helios_r30', 'helios_r50', 'helios_r70', 'helios_w30', 'helios_w50', 'helios_p10', 'helios_p13', 'helios_q10', 'oracle'];
const LABEL: Record<string, string> = {
  naive: 'naive', reactive: 'reactive (baseline)', helios_r30: 'r30 · all-day reserve', helios_r50: 'r50 · all-day reserve',
  helios_r70: 'r70 · all-day reserve', helios_w30: 'w30 · evening window', helios_w50: 'w50 · evening window',
  helios_p10: 'p10 · predictor', helios_p13: 'p13 · predictor', helios_q10: 'q10 · predictor, no top-up', oracle: 'oracle · perfect foresight',
};

export const strategyRank = (s: string) => { const i = ORDER.indexOf(s); return i === -1 ? ORDER.length : i; };
export const byStrategy = (a: string, b: string) => strategyRank(a) - strategyRank(b);
export const sortRows = <T extends { strategy: string }>(rows: T[]): T[] => [...rows].sort((a, b) => byStrategy(a.strategy, b.strategy));
export const label = (s: string) => LABEL[s] ?? s;
export const stratColor = (s: string) => {
  if (s === 'oracle') return '#7c3aed';
  if (s === 'reactive') return '#64748b';
  if (s.startsWith('helios_r')) return '#3b82f6';
  if (s.startsWith('helios_w')) return '#0d9488';
  if (s.startsWith('helios_p') || s.startsWith('helios_q')) return '#d97706';
  return '#94a3b8';
};

const pct = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
const ticks = (min: number, max: number, n = 5) => {
  const raw = (max - min || 1) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
};
const Empty = () => <p className="empty">No data for this selection.</p>;

export const BarChart = ({ data }: { data: Row[] }) => {
  if (!data.length) return <Empty />;
  const rows = sortRows(data);
  const m = (d: Row) => d.saved_pct_mean ?? 0, s = (d: Row) => d.saved_pct_std ?? 0;
  const lo = Math.min(0, ...rows.map(d => m(d) - s(d))), hi = Math.max(0, ...rows.map(d => m(d) + s(d)));
  const pad = (hi - lo) * 0.12 || 1, min = lo - pad, max = hi + pad;
  const W = 720, L = 210, R = 70, T = 12, B = 52, RH = 44, H = T + rows.length * RH + B;
  const x = (v: number) => L + ((v - min) / (max - min)) * (W - L - R);
  return (
    <svg className="svg-container" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mean cost saving versus reactive, by strategy">
      {ticks(min, max).map(t => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={T} y2={H - B} className="grid-line" />
          <text x={x(t)} y={H - B + 20} textAnchor="middle" className="tick">{t}%</text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={T} y2={H - B} className="zero-line" />
      {rows.map((d, i) => {
        const y = T + i * RH + 8, mid = y + 12, pos = m(d) >= 0, edge = pos ? m(d) + s(d) : m(d) - s(d);
        return (
          <g key={d.strategy}>
            <text x={L - 12} y={mid + 5} textAnchor="end" className="row-label">{label(d.strategy)}</text>
            <rect x={Math.min(x(0), x(m(d)))} y={y} width={Math.max(Math.abs(x(m(d)) - x(0)), 1.5)} height={24} rx={4} fill={stratColor(d.strategy)} />
            {s(d) > 0 && <path className="whisker" d={`M${x(m(d) - s(d))} ${mid}H${x(m(d) + s(d))}M${x(m(d) - s(d))} ${mid - 5}v10M${x(m(d) + s(d))} ${mid - 5}v10`} />}
            <text x={x(edge) + (pos ? 8 : -8)} y={mid + 5} textAnchor={pos ? 'start' : 'end'} className="val">{pct(m(d))}</text>
          </g>
        );
      })}
      <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" className="axis-title">Cost saving vs reactive (mean ± 1 std over outage years)</text>
    </svg>
  );
};

export const ScatterChart = ({ data }: { data: Row[] }) => {
  if (!data.length) return <Empty />;
  const rows = sortRows(data);
  const W = 720, H = 420, L = 64, R = 24, T = 20, B = 60;
  const xs = rows.map(d => d.cost_ngn ?? 0), ys = rows.map(d => d.gen_hours ?? 0);
  const span = (a: number[]) => (Math.max(...a) - Math.min(...a)) || 1;
  const x0 = Math.min(...xs) - span(xs) * 0.1, x1 = Math.max(...xs) + span(xs) * 0.1;
  const y0 = Math.max(0, Math.min(...ys) - span(ys) * 0.1), y1 = Math.max(...ys) + span(ys) * 0.1;
  const px = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const py = (v: number) => H - B - ((v - y0) / (y1 - y0)) * (H - T - B);
  let prev = -Infinity;
  const ly: Record<string, number> = {};
  [...rows].sort((a, b) => py(a.gen_hours ?? 0) - py(b.gen_hours ?? 0)).forEach(d => {
    prev = Math.max(py(d.gen_hours ?? 0), prev + 17); ly[d.strategy] = prev;
  });
  return (
    <svg className="svg-container" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Total cost versus generator hours, by strategy">
      {ticks(y0, y1).map(t => (
        <g key={`y${t}`}><line x1={L} x2={W - R} y1={py(t)} y2={py(t)} className="grid-line" /><text x={L - 10} y={py(t) + 4} textAnchor="end" className="tick">{t}</text></g>
      ))}
      {ticks(x0, x1).map(t => (
        <g key={`x${t}`}><line x1={px(t)} x2={px(t)} y1={T} y2={H - B} className="grid-line" /><text x={px(t)} y={H - B + 20} textAnchor="middle" className="tick">₦{Math.round(t / 1000)}k</text></g>
      ))}
      {rows.map(d => {
        const cx = px(d.cost_ngn ?? 0), cy = py(d.gen_hours ?? 0), left = cx > W * 0.7;
        return (
          <g key={d.strategy}>
            <circle cx={cx} cy={cy} r={8} fill={stratColor(d.strategy)} stroke="#fff" strokeWidth={2} />
            <text x={cx + (left ? -14 : 14)} y={ly[d.strategy] + 4} textAnchor={left ? 'end' : 'start'} className="val">{d.strategy.replace('helios_', '')}</text>
          </g>
        );
      })}
      <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" className="axis-title">Total cost per year (₦)</text>
      <text transform={`translate(16 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle" className="axis-title">Generator hours per year</text>
    </svg>
  );
};

const mix = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

export const Heatmap = ({ data }: { data: Cell[] }) => {
  if (!data.length) return <Empty />;
  const pvs = [...new Set(data.map(d => d.pv_kw))].sort((a, b) => a - b);
  const bats = [...new Set(data.map(d => d.battery_kwh))].sort((a, b) => a - b);
  const maxAbs = Math.max(...data.map(d => Math.abs(d.value ?? 0)), 1);
  const L = 96, T = 52, cw = 150, ch = 78, W = L + bats.length * cw + 12, H = T + pvs.length * ch + 12;
  return (
    <svg className="svg-container" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cost saving by PV and battery size">
      <text x={L + (bats.length * cw) / 2} y={16} textAnchor="middle" className="axis-title">Battery capacity</text>
      {bats.map((b, xi) => <text key={b} x={L + xi * cw + cw / 2} y={42} textAnchor="middle" className="tick">{b} kWh</text>)}
      {pvs.map((p, yi) => <text key={p} x={L - 12} y={T + yi * ch + ch / 2 + 5} textAnchor="end" className="row-label">{p} kW PV</text>)}
      {pvs.map((p, yi) => bats.map((b, xi) => {
        const v = data.find(d => d.pv_kw === p && d.battery_kwh === b)?.value;
        const t = v == null ? 0 : Math.min(Math.abs(v) / maxAbs, 1);
        const fill = v == null ? '#e2e8f0' : `rgb(${mix([248, 250, 252], v >= 0 ? [13, 148, 136] : [220, 38, 38], 0.08 + 0.92 * t).join(',')})`;
        return (
          <g key={`${p}-${b}`}>
            <rect x={L + xi * cw + 2} y={T + yi * ch + 2} width={cw - 4} height={ch - 4} rx={8} fill={fill} />
            <text x={L + xi * cw + cw / 2} y={T + yi * ch + ch / 2 + 6} textAnchor="middle" className="val" fill={t > 0.55 ? '#fff' : '#0f172a'}>{v == null ? '–' : pct(v)}</text>
          </g>
        );
      }))}
    </svg>
  );
};
