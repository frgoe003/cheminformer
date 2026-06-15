import Papa from "papaparse";

// ── GPU instance registry ──────────────────────────────────────────────────

export type GpuId = "g7e" | "g6e" | "g5";

export interface GpuInstance {
  id: GpuId;
  instanceType: string;
  gpu: string;
  vram: string;
  vcpus: number;
  ram: string;
  resultsDir: string;
}

export const GPU_INSTANCES: Record<GpuId, GpuInstance> = {
  g7e: {
    id: "g7e",
    instanceType: "g7e.4xlarge",
    gpu: "NVIDIA RTX PRO 6000 Blackwell",
    vram: "96 GB GDDR7",
    vcpus: 16,
    ram: "128 GiB",
    resultsDir: "results_g7e",
  },
  g6e: {
    id: "g6e",
    instanceType: "g6e.4xlarge",
    gpu: "NVIDIA L40S",
    vram: "48 GB GDDR6",
    vcpus: 16,
    ram: "128 GiB",
    resultsDir: "results_g6e",
  },
  g5: {
    id: "g5",
    instanceType: "g5.4xlarge",
    gpu: "NVIDIA A10G",
    vram: "24 GB GDDR6",
    vcpus: 16,
    ram: "64 GiB",
    resultsDir: "results_g5",
  },
};

export const GPU_ORDER: GpuId[] = ["g7e", "g6e", "g5"];

// ── Types ──────────────────────────────────────────────────────────────────

export type Subset = "Small Ligands" | "Large Ligands" | "Pentapeptides" | "Dimers";

export interface PerMolRow {
  name: string;
  n_atoms: number;
  charge: number;
  mae_kcal: number;
  model: string;
  subset: Subset;
}

export interface ResultRow {
  system: string;
  n_atoms: number;
  model: string;
  ms_per_step: number;
  vram_mib: number;
  avg_power_w: number;
  status: string;
}

export interface MaeMatrix {
  models: string[];
  cols: string[];
  counts: Record<string, number>;
  data: Record<string, Record<string, number | null>>;
  n_ok: Record<string, Record<string, number>>;
  min: number;
  max: number;
}

export type PerfMetric = "ms_per_step" | "vram_mib" | "avg_power_w";

export type PerfPoint = { n_atoms: number; system: string; [model: string]: number | string | undefined };

/** For the "by system" grouped bar view: one entry per model with values for each GPU */
export interface SystemPerfRow {
  model: string;
  [gpuId: string]: number | string | undefined;
}

// ── Constants ──────────────────────────────────────────────────────────────

const KJ_TO_KCAL = 1 / 4.184;
const BASE = import.meta.env.BASE_URL;

export const MAE_COLS = [
  "Overall", "Small Ligands", "Large Ligands",
  "Pentapeptides", "Dimers", "Neutral", "Charged",
] as const;

export const COL_LABEL: Record<string, string[]> = {
  "Overall":      ["Overall"],
  "Small Ligands":["Small", "Ligands"],
  "Large Ligands":["Large", "Ligands"],
  "Pentapeptides":["Penta-", "peptides"],
  "Dimers":       ["Dimers"],
  "Neutral":      ["Neutral"],
  "Charged":      ["Charged"],
};

export const COL_COUNT: Record<string, number> = {
  "Overall": 800, "Small Ligands": 200, "Large Ligands": 200,
  "Pentapeptides": 200, "Dimers": 200, "Neutral": 617, "Charged": 183,
};

export const FILE_TO_MODEL: Record<string, string> = {
  "AceFF-1.1":      "AceFF-1.1",
  "AceFF-2.0":      "AceFF-2.0",
  AIMNet2:          "AIMNet2",
  "Egret-1":        "Egret-1",
  "FeNNix-Bio1_M_": "FeNNix-Bio1(M)",
  "FeNNix-Bio1_S_": "FeNNix-Bio1(S)",
  "MACE-MH-1":      "MACE-MH-1",
  "MACE-OFF23_L_":  "MACE-OFF23(L)",
  "MACE-OFF23_S_":  "MACE-OFF23(S)",
  "MACE-OFF24_M_":  "MACE-OFF24(M)",
  "MACE-OMOL-0":    "MACE-OMOL-0",
  "MACELES-OFF":    "MACELES-OFF",
  "Orb-v3-omol":    "Orb-v3-omol",
  "UMA-m-1.1":        "UMA-m-1.1",
  "UMA-s-1.2":        "UMA-s-1.2",
  "AllScAIP-cons":    "AllScAIP-cons",
  "AllScAIP-direct":  "AllScAIP-direct",
  "g-xTB":            "g-xTB",
  "GFN2-xTB":         "GFN2-xTB",
  "polar-1-l":      "polar-1-l",
  "polar-1-m":      "polar-1-m",
  "polar-1-s":      "polar-1-s",
};

export const MODEL_LABEL: Record<string, string> = {
  AIMNet2:           "AIMNet2",
  "AceFF-1.1":       "AceFF-1.1",
  "AceFF-2.0":       "AceFF-2.0",
  "Egret-1":         "Egret-1",
  "FeNNix-Bio1(M)":  "FeNNix-Bio1(M)",
  "FeNNix-Bio1(S)":  "FeNNix-Bio1(S)",
  "MACE-MH-1":       "MACE-MH-1",
  "MACE-OFF23(L)":   "MACE-OFF23(L)",
  "MACE-OFF23(S)":   "MACE-OFF23(S)",
  "MACE-OFF24(M)":   "MACE-OFF24(M)",
  "MACE-OMOL-0":     "MACE-OMOL-0",
  "MACELES-OFF":     "MACELES-OFF",
  "Orb-v3-omol":     "Orb-v3-omol",
  "UMA-m-1.1":       "UMA-m-1.1",
  "UMA-s-1.2":       'UMA-s-1.2 (task="omol")',
  "AllScAIP-cons":   'AllScAIP (cons, task="omol")',
  "AllScAIP-direct": 'AllScAIP (direct, task="omol")',
  "g-xTB":           "g-xTB",
  "GFN2-xTB":        "GFN2-xTB",
  "polar-1-l":       "Polar-1(L)",
  "polar-1-m":       "Polar-1(M)",
  "polar-1-s":       "Polar-1(S)",
};

export const MODEL_COLOR: Record<string, string> = {
  AIMNet2:           "#6366f1",
  "AceFF-1.1":       "#3b82f6",
  "AceFF-2.0":       "#1d4ed8",
  "Egret-1":         "#f59e0b",
  "FeNNix-Bio1(M)":  "#ec4899",
  "FeNNix-Bio1(S)":  "#db2777",
  "MACE-MH-1":       "#8b5cf6",
  "MACE-OFF23(L)":   "#7c3aed",
  "MACE-OFF23(S)":   "#a78bfa",
  "MACE-OFF24(M)":   "#d946ef",
  "MACE-OMOL-0":     "#06b6d4",
  "MACELES-OFF":     "#0e7490",
  "Orb-v3-omol":     "#14b8a6",
  "UMA-m-1.1":       "#f97316",
  "UMA-s-1.2":       "#ea580c",
  "AllScAIP-cons":   "#0284c7",
  "AllScAIP-direct": "#0ea5e9",
  "g-xTB":           "#78716c",
  "GFN2-xTB":        "#a8a29e",
  "polar-1-l":       "#22c55e",
  "polar-1-m":       "#16a34a",
  "polar-1-s":       "#84cc16",
};

export const GPU_COLOR: Record<GpuId, string> = {
  g7e: "#4f46e5",
  g6e: "#0891b2",
  g5: "#ea580c",
};

export const PERF_SYSTEMS = [
  { system: "capped_ala", n_atoms: 22,    label: "capped_ala",  shortLabel: "Ala-22",     trajectoryGif: "data/trajectories/capped_ala.gif" },
  { system: "chignolin",  n_atoms: 138,   label: "chignolin",   shortLabel: "CLN-138",    trajectoryGif: "data/trajectories/chignolin.gif" },
  { system: "ubiquitin",  n_atoms: 602,   label: "ubiquitin",   shortLabel: "Ubq-602",    trajectoryGif: "data/trajectories/ubiquitin.gif" },
  { system: "2LZM",       n_atoms: 1427,  label: "2LZM",        shortLabel: "2LZM-1.4k",  trajectoryGif: "data/trajectories/2LZM.gif" },
  { system: "1ZG4",       n_atoms: 2224,  label: "1ZG4",        shortLabel: "1ZG4-2.2k",  trajectoryGif: "data/trajectories/1ZG4.gif" },
  { system: "3N5G",       n_atoms: 2304,  label: "3N5G",        shortLabel: "3N5G-2.3k",  trajectoryGif: "data/trajectories/3N5G.gif" },
  { system: "5G1P",       n_atoms: 13193, label: "5G1P",        shortLabel: "5G1P-13k",   trajectoryGif: "data/trajectories/5G1P.gif" },
  { system: "1B3B",       n_atoms: 19008, label: "1B3B",        shortLabel: "1B3B-19k",   trajectoryGif: "data/trajectories/1B3B.gif" },
  { system: "9VM6",       n_atoms: 53700, label: "9VM6",        shortLabel: "9VM6-54k",   trajectoryGif: "data/trajectories/9VM6.gif" },
  { system: "water_99k",  n_atoms: 99999, label: "water_99k",   shortLabel: "H₂O-100k",   trajectoryGif: "data/trajectories/water_99k.gif" },
];

export const PERF_METRIC_LABEL: Record<PerfMetric, string> = {
  ms_per_step: "ms / step",
  vram_mib:    "VRAM (MiB)",
  avg_power_w: "Avg Power (W)",
};

// ── Helpers ────────────────────────────────────────────────────────────────

export function getSubset(name: string | number, n_atoms: number): Subset {
  const s = String(name);
  if (s.includes(" ")) return "Dimers";
  if (s.includes("-")) return "Pentapeptides";
  return n_atoms <= 50 ? "Small Ligands" : "Large Ligands";
}

function percentile(arr: number[], p: number) {
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * p)];
}

// ── CSV loader ─────────────────────────────────────────────────────────────

async function fetchCsv<T>(url: string): Promise<T[]> {
  const res = await fetch(url);
  if (!res.ok) return []; // skip missing files (e.g. g6e has no fennix)
  const text = await res.text();
  const { data } = Papa.parse<T>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return data;
}

export async function loadMaePerMol(): Promise<PerMolRow[]> {
  const all: PerMolRow[] = [];
  await Promise.all(
    Object.entries(FILE_TO_MODEL).map(async ([stem, model]) => {
      const rows = await fetchCsv<{
        name: string; n_atoms: number; charge: number; mae_kj_mol: number;
      }>(`${BASE}data/mae_per_mol/${stem}.csv`);
      for (const r of rows) {
        all.push({
          name: String(r.name),
          n_atoms: r.n_atoms,
          charge: r.charge,
          mae_kcal: r.mae_kj_mol * KJ_TO_KCAL,
          model,
          subset: getSubset(r.name, r.n_atoms),
        });
      }
    }),
  );
  return all;
}

export async function loadResults(dir: string): Promise<ResultRow[]> {
  const stems = [
    "results_aceff", "results_aimnet", "results_allscaip", "results_egret", "results_fennix",
    "results_mace", "results_maceles", "results_orb", "results_uma",
  ];
  const all: ResultRow[] = [];
  await Promise.all(
    stems.map(async (stem) => {
      const rows = await fetchCsv<ResultRow>(`${BASE}data/${dir}/${stem}.csv`);
      for (const r of rows) {
        if (r.status === "ok" && typeof r.n_atoms === "number") all.push(r);
      }
    }),
  );
  return all;
}

// ── Aggregation ────────────────────────────────────────────────────────────

export function buildMaeMatrix(rows: PerMolRow[]): MaeMatrix {
  const modelSet = new Set(rows.map((r) => r.model));
  const models = [...modelSet];

  const data: MaeMatrix["data"] = {};
  const n_ok: MaeMatrix["n_ok"] = {};

  for (const model of models) {
    const mr = rows.filter((r) => r.model === model);
    n_ok[model] = {
      "Overall":       mr.length,
      "Small Ligands": mr.filter((r) => r.subset === "Small Ligands").length,
      "Large Ligands": mr.filter((r) => r.subset === "Large Ligands").length,
      "Pentapeptides": mr.filter((r) => r.subset === "Pentapeptides").length,
      "Dimers":        mr.filter((r) => r.subset === "Dimers").length,
      "Neutral":       mr.filter((r) => r.charge === 0).length,
      "Charged":       mr.filter((r) => r.charge !== 0).length,
    };
    const mean = (f: (r: PerMolRow) => boolean) => {
      const vals = mr.filter(f).map((r) => r.mae_kcal).filter((v) => Number.isFinite(v));
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    data[model] = {
      "Overall":       mean(() => true),
      "Small Ligands": mean((r) => r.subset === "Small Ligands"),
      "Large Ligands": mean((r) => r.subset === "Large Ligands"),
      "Pentapeptides": mean((r) => r.subset === "Pentapeptides"),
      "Dimers":        mean((r) => r.subset === "Dimers"),
      "Neutral":       mean((r) => r.charge === 0),
      "Charged":       mean((r) => r.charge !== 0),
    };
  }

  models.sort(
    (a, b) => (data[a]["Overall"] ?? Infinity) - (data[b]["Overall"] ?? Infinity),
  );

  const allVals = models.flatMap((m) =>
    Object.values(data[m]).filter((v): v is number => v !== null && Number.isFinite(v)),
  );
  return {
    models, cols: [...MAE_COLS], counts: COL_COUNT, data, n_ok,
    min: Math.min(...allVals),
    max: percentile(allVals, 0.97),
  };
}

/** Overall MAE sorted ascending — for the bar chart */
export function buildMaeBars(matrix: MaeMatrix): { model: string; mae: number; label: string }[] {
  return matrix.models
    .map((m) => ({ model: m, mae: matrix.data[m]["Overall"] ?? 0, label: MODEL_LABEL[m] ?? m }))
    .sort((a, b) => a.mae - b.mae);
}

/** Line chart data for a single GPU across all systems */
export function buildPerfPoints(
  rows: ResultRow[],
  metric: PerfMetric,
): PerfPoint[] {
  return PERF_SYSTEMS.map(({ system, n_atoms }) => {
    const pt: PerfPoint = { n_atoms, system };
    for (const r of rows.filter((r) => r.system === system)) {
      const v = r[metric];
      if (typeof v === "number" && v > 0) pt[r.model] = v;
    }
    return pt;
  });
}

// ── Model metadata ─────────────────────────────────────────────────────────

export interface ModelMeta {
  elements: number | null;
  params_M: number | null;
  train_M: number | null;
  lot: string;
  charges: boolean | null;
  permissive: boolean | null;
  note?: string;
}

export const MODEL_META: Record<string, ModelMeta> = {
  "AceFF-1.1":      { elements: 12,   params_M: 0.5,   train_M: 11,   lot: "ωB97M-V/def2-TZVPPD",    charges: true,  permissive: true  },
  "AceFF-2.0":      { elements: 12,   params_M: 1.0,   train_M: 12,   lot: "ωB97M-V/def2-TZVPPD",    charges: true,  permissive: true  },
  "AIMNet2":        { elements: 14,   params_M: 2.2,   train_M: 20,   lot: "ωB97M-D3/def2-TZVPP",    charges: true,  permissive: true  },
  "Egret-1":        { elements: 10,   params_M: 3.6,   train_M: 0.95, lot: "ωB97M-D3BJ/def2-TZVPPD", charges: false, permissive: true  },
  "FeNNix-Bio1(S)": { elements: 12,   params_M: 7.4,   train_M: 2.2,  lot: "ωB97M-D3BJ/aug-cc-pVTZ", charges: true,  permissive: false },
  "FeNNix-Bio1(M)": { elements: 12,   params_M: 9.5,   train_M: 2.2,  lot: "ωB97M-D3BJ/aug-cc-pVTZ", charges: true,  permissive: false },
  "MACE-MH-1":      { elements: 89,   params_M: 6.4,   train_M: 116,  lot: "ωB97M-D3BJ/def2-TZVPPD", charges: false, permissive: false },
  "MACE-OFF23(S)":  { elements: 10,   params_M: 0.7,   train_M: 0.95, lot: "ωB97M-D3BJ/def2-TZVPPD", charges: false, permissive: false },
  "MACE-OFF23(L)":  { elements: 10,   params_M: 4.7,   train_M: 0.95, lot: "ωB97M-D3BJ/def2-TZVPPD", charges: false, permissive: false },
  "MACE-OFF24(M)":  { elements: 10,   params_M: 1.4,   train_M: 1.16, lot: "ωB97M-D3BJ/def2-TZVPPD", charges: false, permissive: false },
  "MACE-OMOL-0":    { elements: 89,   params_M: 52,    train_M: 100,  lot: "ωB97M-V/def2-TZVPD",     charges: true,  permissive: false },
  "MACELES-OFF":    { elements: 10,   params_M: 1.9,   train_M: 0.95, lot: "ωB97M-D3BJ/def2-TZVPPD", charges: false, permissive: true  },
  "Orb-v3-omol":    { elements: 89,   params_M: 26,    train_M: 100,  lot: "ωB97M-V/def2-TZVPD",     charges: true,  permissive: true  },
  "UMA-s-1.2":        { elements: 89,   params_M: 150,   train_M: 484,  lot: "ωB97M-V/def2-TZVPD",     charges: true,  permissive: true  },
  "UMA-m-1.1":        { elements: 89,   params_M: 1400,  train_M: 484,  lot: "ωB97M-V/def2-TZVPD",     charges: true,  permissive: true  },
  "AllScAIP-cons":    { elements: 89,   params_M: 102,   train_M: 100,  lot: "ωB97M-V/def2-TZVPD",     charges: true,  permissive: true  },
  "AllScAIP-direct":  { elements: 89,   params_M: 102,   train_M: 100,  lot: "ωB97M-V/def2-TZVPD",     charges: true,  permissive: true  },
  "g-xTB":            { elements: 103,  params_M: null,  train_M: null, lot: "ωB97M-V/def2-TZVPPD",    charges: true,  permissive: null  },
  "GFN2-xTB":         { elements: 86,   params_M: null,  train_M: null, lot: "GFN2-xTB",               charges: true,  permissive: true  },
  "polar-1-s":      { elements: null, params_M: null,  train_M: 100,  lot: "—",                       charges: null,  permissive: false },
  "polar-1-m":      { elements: null, params_M: null,  train_M: 100,  lot: "—",                       charges: null,  permissive: false },
  "polar-1-l":      { elements: null, params_M: null,  train_M: 100,  lot: "—",                       charges: null,  permissive: false },
};

/** Grouped bar data for a single system across multiple GPUs */
export function buildSystemPerfRows(
  perfByGpu: Partial<Record<GpuId, ResultRow[]>>,
  system: string,
  metric: PerfMetric,
): SystemPerfRow[] {
  // collect all models that have data for this system in any GPU
  const modelSet = new Set<string>();
  for (const rows of Object.values(perfByGpu)) {
    if (!rows) continue;
    for (const r of rows) {
      if (r.system === system) modelSet.add(r.model);
    }
  }
  const models = [...modelSet].sort((a, b) => {
    // sort by g7e value ascending (best first), fallback to g6e, then g5
    const va7 = perfByGpu.g7e?.find((r) => r.system === system && r.model === a)?.[metric] ?? Infinity;
    const va6 = perfByGpu.g6e?.find((r) => r.system === system && r.model === a)?.[metric] ?? Infinity;
    const va5 = perfByGpu.g5?.find((r) => r.system === system && r.model === a)?.[metric] ?? Infinity;
    const va = Math.min(va7, va6, va5);

    const vb7 = perfByGpu.g7e?.find((r) => r.system === system && r.model === b)?.[metric] ?? Infinity;
    const vb6 = perfByGpu.g6e?.find((r) => r.system === system && r.model === b)?.[metric] ?? Infinity;
    const vb5 = perfByGpu.g5?.find((r) => r.system === system && r.model === b)?.[metric] ?? Infinity;
    const vb = Math.min(vb7, vb6, vb5);
    return va - vb;
  });

  return models.map((model) => {
    const row: SystemPerfRow = { model };
    for (const [gpuId, rows] of Object.entries(perfByGpu) as [GpuId, ResultRow[]][]) {
      if (!rows) continue;
      const r = rows.find((r) => r.system === system && r.model === model);
      if (r) {
        const v = r[metric];
        if (typeof v === "number" && v > 0) row[gpuId] = v;
      }
    }
    return row;
  });
}
