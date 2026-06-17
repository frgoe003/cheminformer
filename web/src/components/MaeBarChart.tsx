import type { MaeMatrix } from "../data";
import { MODEL_COLOR, buildMaeBars } from "../data";

interface Props {
  matrix: MaeMatrix;
}

export function MaeBarChart({ matrix }: Props) {
  const bars = buildMaeBars(matrix);
  const maxVal = bars[bars.length - 1]?.mae ?? 1;

  return (
    <div className="mae-bars-wrap">
      {bars.map(({ model, mae, label }) => {
        const pct = (mae / maxVal) * 100;
        const color = MODEL_COLOR[model] ?? "#6366f1";
        return (
          <div key={model} className="mae-bar-row">
            <span className="mae-bar-label">{label}</span>
            <div className="mae-bar-track">
              <div
                className="mae-bar-fill"
                style={{ width: `${pct}%`, background: color }}
              />
            </div>
            <span className="mae-bar-val">{mae.toFixed(3)}</span>
          </div>
        );
      })}
      <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: 8, paddingLeft: 150 }}>
        kcal mol⁻¹ — SPICE test set overall MAE 
      </div>
    </div>
  );
}
