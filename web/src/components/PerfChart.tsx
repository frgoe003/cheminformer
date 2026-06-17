import { useState } from "react";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip,
} from "recharts";
import type { PerfMetric, PerfPoint, SystemPerfRow } from "../data";
import { GPU_COLOR, GPU_INSTANCES, GPU_ORDER, MODEL_COLOR, MODEL_LABEL, PERF_METRIC_LABEL, PERF_SYSTEMS } from "../data";

const N_ATOMS_TO_LABEL = Object.fromEntries(
  PERF_SYSTEMS.map(({ n_atoms, label }) => [n_atoms, label]),
);

// ── Axis tick ──────────────────────────────────────────────────────────────

function XTick({ x, y, payload }: { x?: number; y?: number; payload?: { value: number } }) {
  const v = payload?.value ?? 0;
  const label = N_ATOMS_TO_LABEL[v] ?? v.toLocaleString();
  return (
    <g transform={`translate(${x},${y})`}>
      <text dy={13} textAnchor="middle" fill="var(--text-faint)" fontSize={9.5} fontFamily="JetBrains Mono, monospace">
        {label}
      </text>
      <text dy={24} textAnchor="middle" fill="var(--border)" fontSize={8.5} fontFamily="JetBrains Mono, monospace">
        {v >= 1000 ? (v >= 1_000_000 ? `${v / 1_000_000}M` : `${(v / 1000).toFixed(0)}k`) : String(v)}
      </text>
    </g>
  );
}

// ── Tooltip ────────────────────────────────────────────────────────────────

type ScatterPayloadItem = {
  name: string;
  value: number;
  fill: string;
  payload: { x: number; y: number; system: string };
};

function ChartTooltip({
  active, payload, yLabel, yFmt,
}: {
  active?: boolean;
  payload?: ScatterPayloadItem[];
  yLabel: string;
  yFmt: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  const pt = payload[0];
  const systemLabel = N_ATOMS_TO_LABEL[pt.payload.x] ?? pt.payload.system;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__header">{systemLabel}</div>
      <div className="chart-tooltip__row">
        <span className="chart-tooltip__dot" style={{ background: pt.fill }} />
        <span className="chart-tooltip__name">{MODEL_LABEL[pt.name] ?? pt.name}</span>
        <span className="chart-tooltip__val">{yFmt(pt.value)}</span>
      </div>
      <div className="chart-tooltip__footer">{yLabel}</div>
    </div>
  );
}

// ── By-GPU scatter chart ───────────────────────────────────────────────────

const N_ATOMS_TICKS = PERF_SYSTEMS.map((s) => s.n_atoms);

interface ScatterProps {
  data: PerfPoint[];
  metric: PerfMetric;
}

type ScatterPoint = { x: number; y: number; system: string };

function GpuScatterChart({ data, metric }: ScatterProps) {
  const yLabel = PERF_METRIC_LABEL[metric];
  const models = Object.keys(MODEL_COLOR).filter((m) =>
    data.some((pt) => typeof pt[m] === "number"),
  );
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const toggle = (m: string) =>
    setHidden((p) => { const n = new Set(p); n.has(m) ? n.delete(m) : n.add(m); return n; });

  const fmt =
    metric === "ms_per_step"
      ? (v: number) => v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${v.toFixed(0)}ms`
      : metric === "vram_mib"
      ? (v: number) => v >= 1024 ? `${(v / 1024).toFixed(1)} GiB` : `${v.toFixed(0)} MiB`
      : (v: number) => `${v.toFixed(0)} W`;

  const modelPoints: Record<string, ScatterPoint[]> = {};
  for (const model of models) {
    modelPoints[model] = data
      .filter((pt) => typeof pt[model] === "number")
      .map((pt) => ({ x: pt.n_atoms, y: pt[model] as number, system: pt.system }));
  }

  return (
    <div className="perf-chart-wrap">
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 8, right: 24, left: 16, bottom: 36 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis
            dataKey="x"
            scale="log"
            type="number"
            domain={[20, 110000]}
            ticks={N_ATOMS_TICKS}
            tick={<XTick />}
            tickLine={false}
            height={40}
            stroke="var(--chart-axis)"
            name="n_atoms"
          />
          <YAxis
            dataKey="y"
            scale="log"
            type="number"
            domain={["auto", "auto"]}
            tickFormatter={(v) => fmt(v as number)}
            tick={{ fill: "var(--text-faint)", fontSize: 9.5, fontFamily: "JetBrains Mono, monospace" }}
            width={64}
            stroke="var(--chart-axis)"
            name={yLabel}
          />
          <Tooltip
            content={<ChartTooltip yLabel={yLabel} yFmt={fmt} />}
            cursor={{ strokeDasharray: "3 3", stroke: "var(--chart-axis)" }}
          />
          {models.map((model) => (
            <Scatter
              key={model}
              name={model}
              data={modelPoints[model]}
              fill={hidden.has(model) ? "transparent" : (MODEL_COLOR[model] ?? "#888")}
              opacity={hidden.has(model) ? 0 : 1}
              isAnimationActive={false}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>

      <div className="model-legend">
        {models.map((model) => (
          <button
            key={model}
            className={`model-legend__item${hidden.has(model) ? " model-legend__item--off" : ""}`}
            onClick={() => toggle(model)}
          >
            <span className="model-legend__dot" style={{ background: MODEL_COLOR[model] }} />
            {MODEL_LABEL[model] ?? model}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── By-system grouped horizontal bar chart ─────────────────────────────────

interface SystemBarsProps {
  rows: SystemPerfRow[];
  metric: PerfMetric;
}

function SystemBarsChart({ rows, metric }: SystemBarsProps) {
  if (!rows.length) {
    return <div className="empty-notice">No data for this system.</div>;
  }

  let maxVal = 0;
  for (const row of rows) {
    for (const gpuId of GPU_ORDER) {
      const v = row[gpuId];
      if (typeof v === "number" && v > maxVal) maxVal = v;
    }
  }

  const fmt =
    metric === "ms_per_step"
      ? (v: number) => v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${v.toFixed(1)}ms`
      : metric === "vram_mib"
      ? (v: number) => v >= 1024 ? `${(v / 1024).toFixed(1)} GiB` : `${v.toFixed(0)} MiB`
      : (v: number) => `${v.toFixed(0)} W`;

  return (
    <div className="system-bars">
      {rows.map((row) => (
        <div key={row.model} className="system-bar-group">
          <span className="system-bar-model">{MODEL_LABEL[row.model] ?? row.model}</span>
          <div className="system-bar-gpus">
            {GPU_ORDER.map((gpuId) => {
              const v = row[gpuId];
              const hasVal = typeof v === "number";
              const pct = hasVal ? ((v as number) / maxVal) * 100 : 0;
              return (
                <div key={gpuId} className="system-bar-gpu-row">
                  <span className="system-bar-gpu-label">{GPU_INSTANCES[gpuId].instanceType.replace(".4xlarge", "")}</span>
                  <div className="system-bar-track">
                    {hasVal ? (
                      <div
                        className="system-bar-fill"
                        style={{ width: `${pct}%`, background: GPU_COLOR[gpuId] }}
                      />
                    ) : null}
                  </div>
                  <span className="system-bar-val">
                    {hasVal ? fmt(v as number) : <span className="oom-label">OOM</span>}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
      <div style={{ display: "flex", gap: 16, marginTop: 8, paddingLeft: 158 }}>
        {GPU_ORDER.map((gpuId) => (
          <span key={gpuId} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "var(--text-muted)" }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: GPU_COLOR[gpuId], display: "inline-block" }} />
            {GPU_INSTANCES[gpuId].instanceType}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Exports ────────────────────────────────────────────────────────────────

export function PerfLineChart({ data, metric }: ScatterProps) {
  return <GpuScatterChart data={data} metric={metric} />;
}

export function PerfSystemChart({ rows, metric }: SystemBarsProps) {
  return <SystemBarsChart rows={rows} metric={metric} />;
}
