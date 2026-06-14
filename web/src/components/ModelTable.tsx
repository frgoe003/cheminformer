import { useState } from "react";
import type { MaeMatrix, ResultRow, GpuId } from "../data";
import { MODEL_LABEL, MODEL_META, GPU_INSTANCES, GPU_ORDER } from "../data";

const STEPS_PER_NS = 1_000_000;

function nsPerDay(msPerStep: number) {
  return (1000 / msPerStep * 86_400) / STEPS_PER_NS;
}

type SortKey = "model" | "mae" | "nsday" | "vram" | "params" | "train";

interface Row {
  model: string;
  mae:    number | null;
  nsday:  number | null;
  vram:   number | null;
  params: number | null;
  train:  number | null;
  charges:    boolean | null;
  permissive: boolean | null;
}

function fmtNum(v: number | null, dec = 2) {
  if (v === null) return "—";
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return v.toFixed(dec);
}

function SortIcon({ active, dir }: { active: boolean; dir: 1 | -1 }) {
  return (
    <span className="mt-sort-icon" aria-hidden>
      {active ? (dir === 1 ? " ↑" : " ↓") : " ↕"}
    </span>
  );
}

export function ModelTable({
  maeMatrix,
  perfByGpu,
}: {
  maeMatrix: MaeMatrix;
  perfByGpu: Partial<Record<GpuId, ResultRow[]>>;
}) {
  const defaultGpu = (GPU_ORDER.find((id) => perfByGpu[id]) ?? "g7e") as GpuId;
  const [gpu, setGpu]   = useState<GpuId>(defaultGpu);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "mae", dir: 1 });

  const perfRows = perfByGpu[gpu] ?? [];

  const rows: Row[] = maeMatrix.models.map((model) => {
    const mae  = maeMatrix.data[model]?.["Overall"] ?? null;
    const ref  = perfRows.find((r) => r.model === model && r.system === "chignolin");
    const nsday = ref && ref.ms_per_step > 0 ? nsPerDay(ref.ms_per_step) : null;
    const vram  = ref?.vram_mib ?? null;
    const meta  = MODEL_META[model];
    return {
      model,
      mae,
      nsday,
      vram: typeof vram === "number" && vram > 0 ? vram : null,
      params:     meta?.params_M     ?? null,
      train:      meta?.train_M      ?? null,
      charges:    meta?.charges      ?? null,
      permissive: meta?.permissive   ?? null,
    };
  });

  const sorted = [...rows].sort((a, b) => {
    const { key, dir } = sort;
    if (key === "model") return dir * a.model.localeCompare(b.model);
    const av = a[key as keyof Row] as number | null;
    const bv = b[key as keyof Row] as number | null;
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return dir * (av - bv);
  });

  function toggleSort(key: SortKey) {
    setSort((s) => s.key === key ? { key, dir: (s.dir * -1) as 1 | -1 } : { key, dir: 1 });
  }

  function Th({ label, sortKey, title }: { label: string; sortKey: SortKey; title?: string }) {
    return (
      <th className="mt-th mt-th--sortable" onClick={() => toggleSort(sortKey)} title={title}>
        {label}
        <SortIcon active={sort.key === sortKey} dir={sort.dir} />
      </th>
    );
  }

  return (
    <div>
      <div className="hm-topbar" style={{ marginBottom: 12 }}>
        <div className="gpu-selector">
          {GPU_ORDER.filter((id) => perfByGpu[id]).map((id) => (
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
        <span className="hm-hover-hint">speed & VRAM at chignolin (138 atoms)</span>
      </div>

      <div className="hm-scroll">
        <table className="mt-table">
          <thead>
            <tr>
              <Th label="Model"    sortKey="model" />
              <Th label="MAE"      sortKey="mae"   title="Overall SPICE MAE (kcal/mol)" />
              <Th label="ns/day"   sortKey="nsday" title="ns per day at chignolin (1 fs timestep)" />
              <Th label="VRAM"     sortKey="vram"  title="VRAM at chignolin (MiB)" />
              <Th label="Params"   sortKey="params" title="Parameter count (millions)" />
              <Th label="Training" sortKey="train"  title="Training set size (millions)" />
              <th className="mt-th">Charges</th>
              <th className="mt-th">License</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr key={row.model} className="mt-row">
                <td className="mt-td mt-td--model">{MODEL_LABEL[row.model] ?? row.model}</td>
                <td className="mt-td mt-td--num">{row.mae !== null ? row.mae.toFixed(3) : "—"}</td>
                <td className="mt-td mt-td--num">
                  {row.nsday !== null ? (
                    row.nsday >= 10 ? row.nsday.toFixed(1) :
                    row.nsday >= 1  ? row.nsday.toFixed(2) :
                                     row.nsday.toFixed(3)
                  ) : "—"}
                </td>
                <td className="mt-td mt-td--num">
                  {row.vram !== null
                    ? row.vram >= 1024
                      ? `${(row.vram / 1024).toFixed(1)} G`
                      : `${row.vram.toFixed(0)}`
                    : "—"}
                </td>
                <td className="mt-td mt-td--num">{fmtNum(row.params)}</td>
                <td className="mt-td mt-td--num">{fmtNum(row.train)}</td>
                <td className="mt-td mt-td--center">
                  {row.charges === null ? (
                    <span className="mt-badge mt-badge--gray">—</span>
                  ) : row.charges ? (
                    <span className="mt-badge mt-badge--green">✓</span>
                  ) : (
                    <span className="mt-badge mt-badge--gray">—</span>
                  )}
                </td>
                <td className="mt-td mt-td--center">
                  {row.permissive === null ? (
                    <span className="mt-badge mt-badge--gray">—</span>
                  ) : row.permissive ? (
                    <span className="mt-badge mt-badge--green">Open</span>
                  ) : (
                    <span className="mt-badge mt-badge--gray">ASL</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
