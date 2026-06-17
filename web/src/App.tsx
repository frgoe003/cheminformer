import { useEffect, useState } from "react";
import type { MaeMatrix, ResultRow, PerMolRow, GpuId } from "./data";
import { GPU_INSTANCES, buildMaeMatrix, loadMaePerMol, loadResults } from "./data";
import { MaeHeatmap } from "./components/MaeHeatmap";
import { SpeedHeatmap } from "./components/SpeedHeatmap";
import { ModelTable } from "./components/ModelTable";
import { NsdayChart } from "./components/NsdayChart";
import { ParetoScatter } from "./components/ParetoScatter";
import { ModelInfoScatter } from "./components/ModelInfoScatter";
import { SystemTiles } from "./components/SystemTiles";
import { WorkflowDiagram } from "./components/WorkflowDiagram";

function Spinner() { return <div className="spinner" />; }

type Theme = "light" | "dark";
type ThemePreference = Theme | "system";

function getSystemTheme(): Theme {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function getInitialThemePreference(): ThemePreference {
  if (typeof window === "undefined") return "system";
  const stored = window.localStorage.getItem("theme");
  return stored === "light" || stored === "dark" ? stored : "system";
}

const PARETO_SYSTEMS = [
  { system: "chignolin", nAtoms: 138   },
  { system: "1ZG4",      nAtoms: 2224  },
  { system: "1B3B",      nAtoms: 19008 },
] as const;

export default function App() {
  const [maeRows, setMaeRows]     = useState<PerMolRow[] | null>(null);
  const [perfByGpu, setPerfByGpu] = useState<Partial<Record<GpuId, ResultRow[]>>>({});
  const [error, setError]         = useState<string | null>(null);
  const [showStd, setShowStd]     = useState(false);
  const [themePreference, setThemePreference] = useState<ThemePreference>(getInitialThemePreference);
  const [systemTheme, setSystemTheme] = useState<Theme>(getSystemTheme);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setSystemTheme(media.matches ? "dark" : "light");
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  const activeTheme = themePreference === "system" ? systemTheme : themePreference;

  useEffect(() => {
    document.documentElement.dataset.theme = activeTheme;
    document.documentElement.style.colorScheme = activeTheme;
    if (themePreference === "system") {
      window.localStorage.removeItem("theme");
    } else {
      window.localStorage.setItem("theme", themePreference);
    }
  }, [activeTheme, themePreference]);

  useEffect(() => {
    Promise.all([
      loadMaePerMol(),
      loadResults(GPU_INSTANCES.g7e.resultsDir),
      loadResults(GPU_INSTANCES.g6e.resultsDir),
      loadResults(GPU_INSTANCES.g5.resultsDir),
    ])
      .then(([mae, g7e, g6e, g5]) => {
        setMaeRows(mae);
        setPerfByGpu({ g7e, g6e, g5 });
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const maeMatrix: MaeMatrix | null = maeRows ? buildMaeMatrix(maeRows) : null;
  const g7eRows = perfByGpu.g7e ?? [];
  const ready = maeMatrix && g7eRows.length > 0;
  const nextTheme = activeTheme === "dark" ? "light" : "dark";

  return (
    <div className="app">

      {/* ── Nav ──────────────────────────────────────────────────────── */}
      <nav className="nav">
        <span className="nav-logo">
          cheminformer / <strong>MLIP benchmark</strong>
        </span>
        <div className="nav-right">
          <button
            className="theme-toggle"
            type="button"
            aria-pressed={activeTheme === "dark"}
            aria-label={`Switch to ${nextTheme} mode`}
            title={`Switch to ${nextTheme} mode`}
            onClick={() => setThemePreference(nextTheme)}
          >
            <span className="theme-toggle__icon" aria-hidden="true">
              {activeTheme === "dark" ? "☀" : "☾"}
            </span>
            <span>{nextTheme === "dark" ? "Dark" : "Light"}</span>
          </button>
          <a
            className="nav-link"
            href="https://github.com/frgoe003/cheminformer"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub ↗
          </a>
        </div>
      </nav>

      {/* ── Page ─────────────────────────────────────────────────────── */}
      <div className="page-view">
        <div className="page-inner">
          {error && <div className="chart-error">{error}</div>}

          {/* ── 1. Systems ───────────────────────────────────────────── */}
          <section>
            <div className="page-section-label">Benchmark Systems (Speed)</div>
            <SystemTiles />
          </section>


          {/* ── 3. Speed heatmap ───────────────────────────────────────── */}
          <section>
            <div className="page-section-label">Speed Benchmark</div>
            <div className="card">
              <div className="section-subtitle">
                NVT MD simulation on AWS EC2 · 100 warmup + 100 production steps · ms per step
              </div>
              {Object.keys(perfByGpu).length === 0
                ? <div className="chart-loading"><Spinner /> Loading…</div>
                : <SpeedHeatmap rowsByGpu={perfByGpu} />
              }
            </div>
          </section>

          {/* ── 3. MAE heatmap ───────────────────────────────────────── */}
          <section>
            <div className="page-section-label">SPICE Benchmark (Accuracy)</div>
            <div className="card">
              <div className="section-header-row">
                <div className="section-subtitle">
                  Mean absolute error (kcal mol⁻¹) across molecule subsets of the{" "}
                  <a className="subtle-link" href="https://zenodo.org/records/19633352" target="_blank" rel="noopener noreferrer">
                    SPICE test set (v3) ↗
                  </a>.
                </div>
                <button
                  className={`hm-std-toggle${showStd ? " hm-std-toggle--active" : ""}`}
                  onClick={() => setShowStd((s) => !s)}
                  title="Show standard deviation across molecules"
                >
                  ± std
                </button>
              </div>
              {!maeMatrix
                ? <div className="chart-loading"><Spinner /> Loading…</div>
                : <MaeHeatmap matrix={maeMatrix} showStd={showStd} />
              }
            </div>
          </section>



          

          {/* ── Model overview table ─────────────────────────────────── */}
          <section>
            <div className="page-section-label">Model Overview</div>
            <div className="card">
              <div className="section-subtitle">
                All models ranked by overall SPICE MAE · click column headers to sort
              </div>
              {!ready
                ? <div className="chart-loading"><Spinner /> Loading…</div>
                : <ModelTable maeMatrix={maeMatrix!} perfByGpu={perfByGpu} />
              }
            </div>
          </section>

          {/* ── 4. Pareto plots ──────────────────────────────────────── */}
          <section>
            <div className="page-section-label">Accuracy vs Speed</div>
            <div className="page-section-meta">
              SPICE MAE from test set · speed from NVT MD simulation ·
              g7e.4xlarge (RTX PRO 6000 Blackwell, 96 GB) · dashed line = Pareto frontier
            </div>
            {!ready
              ? <div className="chart-loading"><Spinner /> Loading…</div>
              : (
                <div className="card-grid-3">
                  {PARETO_SYSTEMS.map(({ system, nAtoms }) => (
                    <div key={system} className="card">
                      <div className="section-title" style={{ marginBottom: 2 }}>{system}</div>
                      <div className="section-subtitle" style={{ marginBottom: 12 }}>
                        {nAtoms.toLocaleString()} atoms
                      </div>
                      <ParetoScatter
                        system={system}
                        nAtoms={nAtoms}
                        maeMatrix={maeMatrix!}
                        perfRows={g7eRows}
                        showTrajectory={false}
                      />
                    </div>
                  ))}
                </div>
              )
            }
          </section>

          {/* ── ns/day scaling ───────────────────────────────────────── */}
          <section>
            <div className="page-section-label">Speed Scaling</div>
            <div className="card">
              <div className="section-subtitle">
                ns / day vs system size — shows which models scale efficiently to large systems
              </div>
              {Object.keys(perfByGpu).length === 0
                ? <div className="chart-loading"><Spinner /> Loading…</div>
                : <NsdayChart rowsByGpu={perfByGpu} />
              }
            </div>
          </section>

          {/* ── 5. MAE vs model properties ───────────────────────────── */}
          <section>
            <div className="page-section-label">Model Properties vs Accuracy</div>
            {!maeMatrix
              ? <div className="chart-loading"><Spinner /> Loading…</div>
              : (
                <div className="card-grid-2">
                  <div className="card">
                    <div className="section-title" style={{ marginBottom: 12 }}>
                      SPICE MAE vs Parameters
                    </div>
                    <ModelInfoScatter
                      maeMatrix={maeMatrix}
                      xKey="params_M"
                      xLabel="Parameters"
                    />
                  </div>
                  <div className="card">
                    <div className="section-title" style={{ marginBottom: 12 }}>
                      SPICE MAE vs Training Samples
                    </div>
                    <ModelInfoScatter
                      maeMatrix={maeMatrix}
                      xKey="train_M"
                      xLabel="Training Samples"
                    />
                  </div>
                </div>
              )
            }
          </section>

          {/* ── 2. Workflow ──────────────────────────────────────────── */}
          <section>
            <div className="page-section-label">Benchmark Protocol</div>
            <WorkflowDiagram />
          </section>


        </div>
      </div>
    </div>
  );
}
