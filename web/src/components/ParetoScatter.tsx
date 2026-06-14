import {
  CartesianGrid, ComposedChart, Line, ResponsiveContainer,
  Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import { MODEL_COLOR, MODEL_LABEL } from "../data";
import type { MaeMatrix, ResultRow } from "../data";
import { TrajectoryViewer } from "./TrajectoryViewer";

// ── Types ─────────────────────────────────────────────────────────────────

type Pt = { model: string; x: number; y: number; pareto: boolean };

// ── Pareto computation ────────────────────────────────────────────────────

function computePareto(pts: Pt[]): Set<string> {
  const sorted = [...pts].sort((a, b) => a.x - b.x); // fastest first (lowest ms/step)
  const frontier = new Set<string>();
  let bestMAE = Infinity;
  for (const pt of sorted) {
    if (pt.y < bestMAE) {
      frontier.add(pt.model);
      bestMAE = pt.y;
    }
  }
  return frontier;
}

// ── Custom dot for Scatter ────────────────────────────────────────────────

function ModelDot(props: any) {
  const { cx, cy, payload } = props as { cx: number; cy: number; payload: Pt };
  if (cx == null || cy == null || !payload) return null;
  const color = MODEL_COLOR[payload.model] ?? "#888";
  const r = payload.pareto ? 5.5 : 3.5;
  return (
    <g>
      {payload.pareto && (
        <circle cx={cx} cy={cy} r={r + 5} fill={color} opacity={0.1} />
      )}
      <circle
        cx={cx} cy={cy} r={r}
        fill={color}
        stroke={payload.pareto ? "white" : "none"}
        strokeWidth={payload.pareto ? 1.5 : 0}
        opacity={payload.pareto ? 1 : 0.6}
      />
    </g>
  );
}

// ── Tooltip ───────────────────────────────────────────────────────────────

function ScatterTooltip({ active, payload }: { active?: boolean; payload?: any[] }) {
  if (!active || !payload?.length) return null;
  // ComposedChart may deliver scatter payload inside nested structure
  const raw = payload.find((p: any) => p.payload?.model);
  const d: Pt | null = raw?.payload ?? null;
  if (!d) return null;
  const color = MODEL_COLOR[d.model] ?? "#888";
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__row" style={{ marginBottom: 4 }}>
        <span className="chart-tooltip__dot" style={{ background: color }} />
        <span style={{ fontWeight: 600, color: "#111827", fontSize: 12 }}>
          {MODEL_LABEL[d.model] ?? d.model}
        </span>
      </div>
      <div style={{ paddingLeft: 14, lineHeight: 1.8, fontSize: 10.5, color: "#374151" }}>
        <div>
          SPICE MAE{" "}
          <span style={{ fontWeight: 600 }}>{d.y.toFixed(3)}</span>{" "}
          <span style={{ color: "#9ca3af" }}>kcal/mol</span>
        </div>
        <div>
          Speed{" "}
          <span style={{ fontWeight: 600 }}>
            {d.x >= 1000 ? `${(d.x / 1000).toFixed(1)}k` : d.x.toFixed(1)}
          </span>{" "}
          <span style={{ color: "#9ca3af" }}>ms/step</span>
        </div>
        {d.pareto && (
          <div style={{ color: "#4f46e5", marginTop: 2 }}>★ Pareto frontier</div>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────

interface Props {
  system: string;
  nAtoms: number;
  maeMatrix: MaeMatrix;
  perfRows: ResultRow[];
  showTrajectory?: boolean;
}

export function ParetoScatter({ system, nAtoms, maeMatrix, perfRows, showTrajectory }: Props) {
  const sysRows = perfRows.filter((r) => r.system === system);

  const pts: Pt[] = [];
  for (const model of maeMatrix.models) {
    const mae = maeMatrix.data[model]?.["Overall"];
    const row = sysRows.find((r) => r.model === model);
    if (mae != null && row && row.ms_per_step > 0) {
      pts.push({ model, x: row.ms_per_step, y: mae, pareto: false });
    }
  }

  if (!pts.length) {
    return <div className="empty-notice">No data for this system.</div>;
  }

  const paretoSet = computePareto(pts);
  const allPts: Pt[] = pts.map((p) => ({ ...p, pareto: paretoSet.has(p.model) }));
  const paretoLine = allPts
    .filter((p) => p.pareto)
    .sort((a, b) => a.x - b.x);

  const fmtX = (v: number) =>
    v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 10 ? v.toFixed(0) : v.toFixed(1);

  const xs = allPts.map((p) => p.x);
  const xMin = Math.min(...xs) * 0.6;
  const xMax = Math.max(...xs) * 1.6;

  return (
    <div>
      <div style={{ position: "relative" }}>
        {showTrajectory && (
          <div style={{
            position: "absolute", top: 0, right: 0, zIndex: 5,
            width: 150, height: 150,
            borderRadius: 8, overflow: "hidden",
            border: "1px solid #e5e7eb",
            boxShadow: "0 1px 6px rgba(0,0,0,0.07)",
          }}>
            <TrajectoryViewer height={150} showControls={false} />
          </div>
        )}

        <ResponsiveContainer width="100%" height={270}>
          <ComposedChart margin={{ top: 8, right: showTrajectory ? 162 : 16, bottom: 32, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis
              dataKey="x"
              scale="log"
              type="number"
              domain={[xMin, xMax]}
              tickFormatter={fmtX}
              tick={{ fill: "#9ca3af", fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
              stroke="#e5e7eb"
              height={40}
              label={{ value: "ms / step  (log scale)", position: "insideBottom", offset: -6, fill: "#9ca3af", fontSize: 9.5, fontFamily: "JetBrains Mono, monospace" }}
              name="ms/step"
            />
            <YAxis
              dataKey="y"
              type="number"
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => v.toFixed(2)}
              tick={{ fill: "#9ca3af", fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
              stroke="#e5e7eb"
              width={52}
              label={{ value: "↓ SPICE MAE (kcal/mol)", angle: -90, position: "insideLeft", fill: "#9ca3af", fontSize: 9.5, fontFamily: "JetBrains Mono, monospace", offset: 8 }}
              name="SPICE MAE (kcal/mol)"
            />
            <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: "3 3", stroke: "#e5e7eb" }} />

            {/* Pareto frontier line — drawn first so dots sit on top */}
            <Line
              data={paretoLine}
              dataKey="y"
              type="linear"
              dot={false}
              activeDot={false}
              stroke="#4f46e5"
              strokeWidth={1.5}
              strokeDasharray="5 3"
              opacity={0.65}
              legendType="none"
              isAnimationActive={false}
            />

            {/* All model dots */}
            <Scatter
              data={allPts}
              shape={<ModelDot />}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Pareto frontier badge row */}
      {paretoLine.length > 0 && (
        <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: "4px 6px", alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "#9ca3af", marginRight: 2 }}>Frontier:</span>
          {[...paretoLine].sort((a, b) => b.x - a.x).map((p) => (
            <span
              key={p.model}
              style={{
                display: "inline-flex", alignItems: "center", gap: 4,
                padding: "2px 8px", borderRadius: 999,
                background: "#f3f4f6", border: "1px solid #e5e7eb",
                fontSize: 10, fontFamily: "JetBrains Mono, monospace", color: "#374151",
              }}
            >
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: MODEL_COLOR[p.model], flexShrink: 0 }} />
              {MODEL_LABEL[p.model] ?? p.model}
            </span>
          ))}
        </div>
      )}

      <div style={{ marginTop: 6, fontSize: 10, color: "#9ca3af" }}>
        {allPts.length} models · {nAtoms.toLocaleString()} atoms
      </div>
    </div>
  );
}
