import { useState } from "react";
import {
  CartesianGrid, ComposedChart, Line, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { ResultRow, GpuId } from "../data";
import { PERF_SYSTEMS, MODEL_LABEL, MODEL_COLOR, GPU_INSTANCES, GPU_ORDER } from "../data";

const STEPS_PER_NS = 1_000_000;

function nsPerDay(msPerStep: number) {
  return (1000 / msPerStep * 86_400) / STEPS_PER_NS;
}

function fmtAtoms(v: number) {
  if (v >= 1_000) return `${(v / 1000).toFixed(0)}k`;
  return `${v}`;
}

function fmtNsDay(v: number) {
  if (v >= 100) return `${v.toFixed(0)}`;
  if (v >= 10)  return `${v.toFixed(1)}`;
  if (v >= 1)   return `${v.toFixed(2)}`;
  return `${v.toFixed(3)}`;
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: any[]; label?: number }) {
  if (!active || !payload?.length || label == null) return null;
  const sys = PERF_SYSTEMS.find((s) => s.n_atoms === label);
  const entries = payload
    .filter((p) => p.value != null)
    .sort((a, b) => b.value - a.value);
  return (
    <div className="chart-tooltip" style={{ minWidth: 200 }}>
      <div className="chart-tooltip__header">
        {sys?.label ?? label} · {label.toLocaleString()} atoms
      </div>
      {entries.map((p) => (
        <div key={p.dataKey} className="chart-tooltip__row">
          <span className="chart-tooltip__dot" style={{ background: p.color }} />
          <span style={{ flex: 1 }}>{MODEL_LABEL[p.dataKey] ?? p.dataKey}</span>
          <span style={{ fontWeight: 600, color: "var(--text-strong)" }}>{fmtNsDay(p.value)} ns/day</span>
        </div>
      ))}
    </div>
  );
}

export function NsdayChart({ rowsByGpu }: { rowsByGpu: Partial<Record<GpuId, ResultRow[]>> }) {
  const defaultGpu = (GPU_ORDER.find((id) => rowsByGpu[id]) ?? "g7e") as GpuId;
  const [gpu, setGpu] = useState<GpuId>(defaultGpu);

  const rows = rowsByGpu[gpu] ?? [];

  const models = [...new Set(rows.map((r) => r.model))];

  const pts = PERF_SYSTEMS.map(({ system, n_atoms }) => {
    const pt: Record<string, number | string> = { n_atoms, system };
    for (const r of rows) {
      if (r.system === system && r.ms_per_step > 0) {
        pt[r.model] = nsPerDay(r.ms_per_step);
      }
    }
    return pt;
  });

  return (
    <div className="hm-wrap">
      <div className="hm-topbar">
        <div className="gpu-selector">
          {GPU_ORDER.filter((id) => rowsByGpu[id]).map((id) => (
            <button
              key={id}
              className={`gpu-btn${gpu === id ? " gpu-btn--active" : ""}`}
              onClick={() => setGpu(id)}
            >
              {GPU_INSTANCES[id].instanceType}
              <span className="gpu-btn__sub">{GPU_INSTANCES[id].gpu.replace("NVIDIA ", "")}</span>
            </button>
          ))}
        </div>
        <span className="hm-hover-hint">dt = 1 fs</span>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={pts} margin={{ top: 8, right: 24, bottom: 40, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis
            dataKey="n_atoms"
            scale="log"
            type="number"
            domain={[15, 150_000]}
            tickFormatter={fmtAtoms}
            ticks={PERF_SYSTEMS.map((s) => s.n_atoms)}
            tick={{ fill: "var(--text-faint)", fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
            stroke="var(--chart-axis)"
            height={44}
            label={{
              value: "System size (atoms, log scale)",
              position: "insideBottom", offset: -10,
              fill: "var(--text-faint)", fontSize: 9.5, fontFamily: "JetBrains Mono, monospace",
            }}
          />
          <YAxis
            scale="log"
            type="number"
            domain={["auto", "auto"]}
            tickFormatter={fmtNsDay}
            tick={{ fill: "var(--text-faint)", fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
            stroke="var(--chart-axis)"
            width={48}
            label={{
              value: "↑ ns / day (log scale)",
              angle: -90, position: "insideLeft",
              fill: "var(--text-faint)", fontSize: 9.5,
              fontFamily: "JetBrains Mono, monospace", offset: 12,
            }}
            allowDataOverflow
          />
          <Tooltip content={<ChartTooltip />} />
          {models.map((model) => (
            <Line
              key={model}
              dataKey={model}
              type="monotone"
              stroke={MODEL_COLOR[model] ?? "#888"}
              strokeWidth={1.5}
              dot={{ r: 2.5, fill: MODEL_COLOR[model] ?? "#888", strokeWidth: 0 }}
              activeDot={{ r: 4, strokeWidth: 0 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
