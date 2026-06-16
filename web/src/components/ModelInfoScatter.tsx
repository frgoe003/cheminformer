import {
  CartesianGrid, ComposedChart, ResponsiveContainer,
  Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import { MODEL_COLOR, MODEL_LABEL, MODEL_META } from "../data";
import type { MaeMatrix } from "../data";

type Pt = { model: string; x: number; y: number };

function LabeledDot(props: any) {
  const { cx, cy, payload } = props as { cx: number; cy: number; payload: Pt };
  if (cx == null || cy == null || !payload) return null;
  const color = MODEL_COLOR[payload.model] ?? "#888";
  return (
    <g>
      <circle cx={cx} cy={cy} r={4.5} fill={color} opacity={0.85} stroke="var(--surface)" strokeWidth={1} />
      <text
        x={cx + 7} y={cy + 3.5}
        fontSize={8.5}
        fontFamily="JetBrains Mono, monospace"
        fill="var(--text-muted)"
      >
        {MODEL_LABEL[payload.model] ?? payload.model}
      </text>
    </g>
  );
}

function InfoTooltip({ active, payload, xLabel }: { active?: boolean; payload?: any[]; xLabel: string }) {
  if (!active || !payload?.length) return null;
  const raw = payload.find((p: any) => p.payload?.model);
  const d: Pt | null = raw?.payload ?? null;
  if (!d) return null;
  const color = MODEL_COLOR[d.model] ?? "#888";
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__row" style={{ marginBottom: 4 }}>
        <span className="chart-tooltip__dot" style={{ background: color }} />
        <span style={{ fontWeight: 600, color: "var(--text-strong)", fontSize: 12 }}>
          {MODEL_LABEL[d.model] ?? d.model}
        </span>
      </div>
      <div style={{ paddingLeft: 14, lineHeight: 1.8, fontSize: 10.5, color: "var(--text)" }}>
        <div>
          SPICE MAE{" "}
          <span style={{ fontWeight: 600 }}>{d.y.toFixed(3)}</span>{" "}
          <span style={{ color: "var(--text-faint)" }}>kcal/mol</span>
        </div>
        <div>
          {xLabel}{" "}
          <span style={{ fontWeight: 600 }}>{d.x >= 1000 ? `${(d.x / 1000).toFixed(1)}k` : d.x}</span>
          <span style={{ color: "var(--text-faint)" }}> M</span>
        </div>
      </div>
    </div>
  );
}

interface Props {
  maeMatrix: MaeMatrix;
  xKey: "params_M" | "train_M";
  xLabel: string;
}

export function ModelInfoScatter({ maeMatrix, xKey, xLabel }: Props) {
  const pts: Pt[] = [];
  for (const model of maeMatrix.models) {
    const mae = maeMatrix.data[model]?.["Overall"];
    const meta = MODEL_META[model];
    const xVal = meta?.[xKey];
    if (mae != null && xVal != null) {
      pts.push({ model, x: xVal, y: mae });
    }
  }

  if (!pts.length) return null;

  const fmtX = (v: number) =>
    v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 10 ? v.toFixed(0) : v.toFixed(2);

  const xs = pts.map((p) => p.x);
  const xMin = Math.min(...xs) * 0.55;
  const xMax = Math.max(...xs) * 1.8;

  const xTicks = (() => {
    const lo = Math.floor(Math.log10(xMin));
    const hi = Math.ceil(Math.log10(xMax));
    const seen = new Set<number>();
    const out: number[] = [];
    for (let e = lo; e <= hi; e++) {
      for (const m of [1, 2, 5]) {
        const v = m * Math.pow(10, e);
        if (v >= xMin && v <= xMax && !seen.has(v)) { seen.add(v); out.push(v); }
      }
    }
    return out.sort((a, b) => a - b);
  })();

  const makeTooltip = (props: any) => <InfoTooltip {...props} xLabel={xLabel} />;

  return (
    <ResponsiveContainer width="100%" height={290}>
      <ComposedChart margin={{ top: 12, right: 100, bottom: 36, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
        <XAxis
          dataKey="x"
          scale="log"
          type="number"
          domain={[xMin, xMax]}
          ticks={xTicks}
          tickFormatter={fmtX}
          tick={{ fill: "var(--text-faint)", fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
          stroke="var(--chart-axis)"
          height={40}
          label={{
            value: `${xLabel} (M, log scale)`,
            position: "insideBottom", offset: -8,
            fill: "var(--text-faint)", fontSize: 9.5, fontFamily: "JetBrains Mono, monospace",
          }}
          name={xLabel}
        />
        <YAxis
          dataKey="y"
          type="number"
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => v.toFixed(2)}
          tick={{ fill: "var(--text-faint)", fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
          stroke="var(--chart-axis)"
          width={52}
          label={{
            value: "↓ SPICE MAE (kcal/mol)", angle: -90,
            position: "insideLeft", fill: "var(--text-faint)", fontSize: 9.5,
            fontFamily: "JetBrains Mono, monospace", offset: 8,
          }}
          name="SPICE MAE (kcal/mol)"
        />
        <Tooltip content={makeTooltip} cursor={{ strokeDasharray: "3 3", stroke: "var(--chart-axis)" }} />
        <Scatter data={pts} shape={<LabeledDot />} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
