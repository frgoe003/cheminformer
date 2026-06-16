import { useState } from "react";
import type { MaeMatrix } from "../data";
import { COL_LABEL, MODEL_LABEL } from "../data";

function cellBg(val: number, min: number, max: number) {
  const t = Math.max(0, Math.min(1, (val - min) / Math.max(max - min, 0.001)));
  const hue = Math.round(120 * (1 - t));
  return `hsl(${hue}, var(--heatmap-cell-saturation), var(--heatmap-cell-lightness))`;
}

interface TooltipData {
  model: string; col: string; value: number; std: number | null;
  n_ok: number; n_expected: number;
}

function Tooltip({ data, pos, showStd }: {
  data: TooltipData; pos: { x: number; y: number }; showStd: boolean;
}) {
  const nFailed = data.n_expected - data.n_ok;
  return (
    <div className="hm-tooltip" style={{ left: pos.x + 12, top: pos.y - 10 }}>
      <span className="hm-tooltip__model">{MODEL_LABEL[data.model] ?? data.model}</span>
      <span className="hm-tooltip__col">{data.col}</span>
      <span className="hm-tooltip__val">
        {data.value.toFixed(3)} kcal mol⁻¹
        {showStd && data.std !== null && (
          <span className="hm-tooltip__std"> ± {data.std.toFixed(3)}</span>
        )}
      </span>
      {nFailed > 0 && (
        <span className="hm-tooltip__warn">
          {data.n_ok}/{data.n_expected} molecules succeeded ({nFailed} failed)
        </span>
      )}
    </div>
  );
}

export function MaeHeatmap({ matrix, showStd }: { matrix: MaeMatrix; showStd: boolean }) {
  const { models, cols, counts, data, std_data, n_ok, min, max } = matrix;
  const [hover, setHover] = useState<{ data: TooltipData; pos: { x: number; y: number } } | null>(null);
  const SEPARATOR_AFTER = "Dimers";

  return (
    <div className="hm-wrap">
      <div className="hm-scroll">
        <table className="hm-table">
          <thead>
            <tr>
              <th className="hm-th hm-th--model">Model</th>
              {cols.map((col) => (
                <th
                  key={col}
                  className={`hm-th hm-th--col${col === SEPARATOR_AFTER ? " hm-sep-right" : ""}`}
                >
                  {(COL_LABEL[col] ?? [col]).map((line, i) => (
                    <span key={i} className="hm-th__line">{line}</span>
                  ))}
                  <span className="hm-th__count">n={counts[col]}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model} className="hm-row">
                <td className="hm-td hm-td--model">{MODEL_LABEL[model] ?? model}</td>
                {cols.map((col) => {
                  const val         = data[model][col];
                  const std         = std_data[model]?.[col] ?? null;
                  const colOk       = n_ok[model]?.[col] ?? counts[col];
                  const colExpected = counts[col];
                  const nFailed     = colExpected - colOk;
                  const showAsterisk = nFailed > 0 && val !== null;
                  return (
                    <td
                      key={col}
                      className={`hm-td hm-td--val${col === SEPARATOR_AFTER ? " hm-sep-right" : ""}${showStd ? " hm-td--wide" : ""}`}
                      style={val !== null ? { background: cellBg(val, min, max) } : {}}
                      onMouseEnter={(e) =>
                        val !== null &&
                        setHover({
                          data: { model, col, value: val, std, n_ok: colOk, n_expected: colExpected },
                          pos: { x: e.clientX, y: e.clientY },
                        })
                      }
                      onMouseMove={(e) =>
                        hover && setHover((h) => h && { ...h, pos: { x: e.clientX, y: e.clientY } })
                      }
                      onMouseLeave={() => setHover(null)}
                    >
                      {val !== null ? (
                        <>
                          {val.toFixed(2)}
                          {showStd && std !== null && (
                            <span className="hm-cell-std"> ±{std.toFixed(2)}</span>
                          )}
                          {showAsterisk && <sup className="hm-asterisk">*</sup>}
                        </>
                      ) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="hm-legend">
        <span>MAE (kcal mol⁻¹)</span>
        <span className="hm-legend__bar" />
        <span className="hm-legend__range">
          {min.toFixed(2)} – {max.toFixed(2)} kcal mol⁻¹
        </span>
      </div>

      {hover && <Tooltip data={hover.data} pos={hover.pos} showStd={showStd} />}
    </div>
  );
}
