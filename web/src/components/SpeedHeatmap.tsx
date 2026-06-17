import { useState } from "react";
import type { ResultRow, GpuId } from "../data";
import { PERF_SYSTEMS, MODEL_LABEL, GPU_INSTANCES, GPU_ORDER } from "../data";

// AWS prices, US East 1 (N. Virginia), Linux, as of 15 Jun 2026
const GPU_PRICE_PER_HR: Record<GpuId, number> = {
  g7e: 4.02616,
  g6e: 3.00424,
  g5:  1.624,
};
const GPU_SPOT_PRICE_PER_HR: Record<GpuId, number> = {
  g7e: 1.6789,
  g6e: 1.5967,
  g5:  0.7107,
};
const STEPS_PER_NS = 1_000_000; // 1 fs timestep (ASE units.fs)

type Metric = "ms_per_step" | "vram_mib";

function nsPerDay(msPerStep: number) {
  return (1000 / msPerStep * 86_400) / STEPS_PER_NS;
}

function costFor100ns(msPerStep: number, pricePerHr: number) {
  return (100 / nsPerDay(msPerStep)) * 24 * pricePerHr;
}

function fmtNsDay(v: number) {
  if (v >= 100) return `${v.toFixed(0)} ns/day`;
  if (v >= 10)  return `${v.toFixed(1)} ns/day`;
  if (v >= 1)   return `${v.toFixed(2)} ns/day`;
  return `${v.toFixed(3)} ns/day`;
}

function fmtCost(usd: number) {
  if (usd >= 10_000) return `$${(usd / 1000).toFixed(0)}k`;
  if (usd >= 1_000)  return `$${usd.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  if (usd >= 10)     return `$${usd.toFixed(0)}`;
  return `$${usd.toFixed(2)}`;
}

function fmtMs(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}`;
}

function fmtVram(mib: number) {
  return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${mib.toFixed(0)} MiB`;
}

function fmtCell(val: number, metric: Metric) {
  return metric === "vram_mib" ? fmtVram(val) : fmtMs(val);
}

function cellBg(val: number, min: number, max: number) {
  const t = Math.max(0, Math.min(1, (val - min) / Math.max(max - min, 0.001)));
  const hue = Math.round(120 * (1 - t));
  return `hsl(${hue}, var(--heatmap-cell-saturation), var(--heatmap-cell-lightness))`;
}

function buildMatrix(rows: ResultRow[], metric: Metric) {
  const modelSet = new Set<string>();
  for (const r of rows) modelSet.add(r.model);
  const models = [...modelSet];

  const data: Record<string, Record<string, number | null>> = {};
  for (const model of models) {
    data[model] = {};
    for (const { system } of PERF_SYSTEMS) {
      const r = rows.find((r) => r.model === model && r.system === system);
      const v = r?.[metric];
      data[model][system] = typeof v === "number" && v > 0 ? v : null;
    }
  }

  models.sort((a, b) => {
    const med = (m: string) => {
      const vals = PERF_SYSTEMS
        .map(({ system }) => data[m][system])
        .filter((v): v is number => v !== null)
        .sort((x, y) => x - y);
      return vals.length ? vals[Math.floor(vals.length / 2)] : Infinity;
    };
    return med(a) - med(b);
  });

  const allVals = models.flatMap((m) =>
    PERF_SYSTEMS.map(({ system }) => data[m][system]).filter((v): v is number => v !== null),
  );
  const sorted = [...allVals].sort((a, b) => a - b);
  return {
    models, data,
    min: sorted[0] ?? 0,
    max: sorted[Math.floor(sorted.length * 0.95)] ?? 1,
  };
}

interface TooltipData { model: string; system: string; nAtoms: number; value: number }

function SpeedTooltip({ data, pos, gpu }: { data: TooltipData; pos: { x: number; y: number }; gpu: GpuId }) {
  const nd  = nsPerDay(data.value);
  const cod = costFor100ns(data.value, GPU_PRICE_PER_HR[gpu]);
  const cos = costFor100ns(data.value, GPU_SPOT_PRICE_PER_HR[gpu]);
  return (
    <div className="speed-tooltip" style={{ left: pos.x + 16, top: pos.y - 16 }}>
      <div className="speed-tooltip__header">
        <span className="speed-tooltip__model">{MODEL_LABEL[data.model] ?? data.model}</span>
        <span className="speed-tooltip__system">{data.system} · {data.nAtoms.toLocaleString()} atoms</span>
      </div>
      <div className="speed-tooltip__metrics">
        <div className="speed-tooltip__row">
          <span className="speed-tooltip__label">ms / step</span>
          <span className="speed-tooltip__value">{data.value.toFixed(2)}</span>
        </div>
        <div className="speed-tooltip__row">
          <span className="speed-tooltip__label">ns / day</span>
          <span className="speed-tooltip__value">{fmtNsDay(nd)}</span>
        </div>
      </div>
      <div className="speed-tooltip__divider" />
      <div className="speed-tooltip__section-label">100 ns on {GPU_INSTANCES[gpu].instanceType}</div>
      <div className="speed-tooltip__metrics">
        <div className="speed-tooltip__row">
          <span className="speed-tooltip__label">On-demand</span>
          <span className="speed-tooltip__value speed-tooltip__value--cost">{fmtCost(cod)}</span>
        </div>
        <div className="speed-tooltip__row">
          <span className="speed-tooltip__label">Spot</span>
          <span className="speed-tooltip__value speed-tooltip__value--spot">{fmtCost(cos)}</span>
        </div>
      </div>
      <div className="speed-tooltip__footnote">prices as of 15 Jun 2026 · US East 1 (N. Virginia)</div>
    </div>
  );
}

function VramTooltip({ data, pos }: { data: TooltipData; pos: { x: number; y: number } }) {
  return (
    <div className="speed-tooltip" style={{ left: pos.x + 16, top: pos.y - 16 }}>
      <div className="speed-tooltip__header">
        <span className="speed-tooltip__model">{MODEL_LABEL[data.model] ?? data.model}</span>
        <span className="speed-tooltip__system">{data.system} · {data.nAtoms.toLocaleString()} atoms</span>
      </div>
      <div className="speed-tooltip__metrics">
        <div className="speed-tooltip__row">
          <span className="speed-tooltip__label">VRAM</span>
          <span className="speed-tooltip__value">{fmtVram(data.value)}</span>
        </div>
      </div>
    </div>
  );
}

export function SpeedHeatmap({ rowsByGpu }: { rowsByGpu: Partial<Record<GpuId, ResultRow[]>> }) {
  const defaultGpu = (GPU_ORDER.find((id) => rowsByGpu[id]) ?? "g7e") as GpuId;
  const [gpu, setGpu]       = useState<GpuId>(defaultGpu);
  const [metric, setMetric] = useState<Metric>("ms_per_step");
  const [hover, setHover]   = useState<{ data: TooltipData; pos: { x: number; y: number } } | null>(null);

  const rows = rowsByGpu[gpu] ?? [];
  const { models, data, min, max } = buildMatrix(rows, metric);

  return (
    <div className="hm-wrap">
      <div className="hm-topbar">
        <div className="hm-topbar__left">
          <div className="gpu-selector">
            {GPU_ORDER.filter((id) => rowsByGpu[id]).map((id) => (
              <button
                key={id}
                className={`gpu-btn${gpu === id ? " gpu-btn--active" : ""}`}
                onClick={() => { setGpu(id); setHover(null); }}
              >
                {GPU_INSTANCES[id].instanceType}
                <span className="gpu-btn__sub">{GPU_INSTANCES[id].gpu.replace("NVIDIA ", "")}</span>
              </button>
            ))}
          </div>
          <div className="metric-toggle">
            {(["ms_per_step", "vram_mib"] as Metric[]).map((m) => (
              <button
                key={m}
                className={`metric-btn${metric === m ? " metric-btn--active" : ""}`}
                onClick={() => { setMetric(m); setHover(null); }}
              >
                {m === "ms_per_step" ? "ms / step" : "VRAM"}
              </button>
            ))}
          </div>
        </div>
        {metric === "ms_per_step" && (
          <span className="hm-hover-hint">hover for cost estimate</span>
        )}
      </div>

      <div className="hm-scroll">
        <table className="hm-table">
          <thead>
            <tr>
              <th className="hm-th hm-th--model">Model</th>
              {PERF_SYSTEMS.map(({ system, shortLabel }) => (
                <th key={system} className="hm-th hm-th--col">
                  {shortLabel.split("-").map((line, i) => (
                    <span key={i} className="hm-th__line">{line}</span>
                  ))}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model} className="hm-row">
                <td className="hm-td hm-td--model">{MODEL_LABEL[model] ?? model}</td>
                {PERF_SYSTEMS.map(({ system, n_atoms }) => {
                  const val = data[model][system];
                  return (
                    <td
                      key={system}
                      className="hm-td hm-td--val"
                      style={val !== null ? { background: cellBg(val, min, max) } : {}}
                      onMouseEnter={(e) =>
                        val !== null &&
                        setHover({
                          data: { model, system, nAtoms: n_atoms, value: val },
                          pos: { x: e.clientX, y: e.clientY },
                        })
                      }
                      onMouseMove={(e) =>
                        hover && setHover((h) => h && { ...h, pos: { x: e.clientX, y: e.clientY } })
                      }
                      onMouseLeave={() => setHover(null)}
                    >
                      {val !== null ? fmtCell(val, metric) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="hm-legend">
        <span>{metric === "ms_per_step" ? "ms / step" : "VRAM"}</span>
        <span className="hm-legend__bar" />
        <span className="hm-legend__range">
          {metric === "ms_per_step"
            ? `${fmtMs(min)} – ${fmtMs(max)}`
            : `${fmtVram(min)} – ${fmtVram(max)}`}
        </span>
      </div>

      {hover && metric === "ms_per_step" && (
        <SpeedTooltip data={hover.data} pos={hover.pos} gpu={gpu} />
      )}
      {hover && metric === "vram_mib" && (
        <VramTooltip data={hover.data} pos={hover.pos} />
      )}
    </div>
  );
}
