const ENVS = [
  {
    name: "ase-mace",
    models: ["MACE-OFF23(S/L)", "MACE-OFF24(M)", "MACE-MH-1", "MACE-OMOL-0", "Polar-1(S/M/L)"],
    packages: ["mace-torch", "cuequivariance_torch", "openff-toolkit"],
  },
  {
    name: "ase-maceles",
    models: ["MACELES-OFF"],
    packages: ["mace-torch", "cuequivariance_torch", "les"],
  },
  {
    name: "ase-egret",
    models: ["Egret-1"],
    packages: ["mace-torch", "cuequivariance_torch"],
  },
  {
    name: "ase-aceff",
    models: ["AceFF-1.1", "AceFF-2.0"],
    packages: ["torchmd-net", "huggingface_hub", "openff-toolkit", "rdkit"],
  },
  {
    name: "ase-uma",
    models: ["UMA-s-1", "UMA-m-1"],
    packages: ["fairchem-core"],
  },
  {
    name: "ase-aimnet",
    models: ["AIMNet2"],
    packages: ["aimnet[ase]"],
  },
  {
    name: "ase-fennix",
    models: ["FeNNix-Bio1(S)", "FeNNix-Bio1(M)"],
    packages: ["fennol[cuda]"],
  },
  {
    name: "ase-orb",
    models: ["Orb-v3-omol"],
    packages: ["orb-models", "pynanoflann"],
  },
];

export function WorkflowDiagram() {
  return (
    <div className="workflow-wrap">
      <div className="workflow-track">
        <div className="workflow-track-label workflow-track-label--acc">Accuracy</div>
        <WorkflowBox
          icon={<DataIcon />}
          title="SPICE Test Set"
          sub="800 structures · 4 subsets"
          color="#4f46e5"
        />
        <Arrow />
        <WorkflowBox
          icon={<ModelIcon />}
          title="MLIP Inference"
          sub="15+ models"
          color="#0891b2"
        />
        <Arrow />
        <WorkflowBox
          icon={<MetricIcon />}
          title="SPICE MAE"
          sub="kcal/mol per subset"
          color="#059669"
          isOutput
        />
      </div>

      <div className="workflow-track">
        <div className="workflow-track-label workflow-track-label--spd">Speed</div>
        <WorkflowBox
          icon={<ProteinIcon />}
          title="MD Systems"
          sub="22 – 100k atoms"
          color="#4f46e5"
        />
        <Arrow />
        <WorkflowBox
          icon={<SimIcon />}
          title="NVT MD"
          sub="OpenFF charges · 300 K · dt = 1 fs"
          color="#0891b2"
        />
        <Arrow />
        <WorkflowBox
          icon={<SpeedIcon />}
          title="ms / step"
          sub="VRAM · avg power"
          color="#059669"
          isOutput
        />
      </div>

      <details className="wf-env-details">
        <summary className="wf-env-summary">
          <span>Software environment</span>
          <span className="wf-env-base">Python 3.11 · PyTorch cu128 (CUDA 12.8) · ASE</span>
        </summary>
        <div className="wf-env-table">
          {ENVS.map(({ name, models, packages }) => (
            <div key={name} className="wf-env-row">
              <span className="wf-env-name">{name}</span>
              <span className="wf-env-models">{models.join(", ")}</span>
              <span className="wf-env-packages">
                {packages.map((p) => (
                  <span key={p} className="wf-env-pkg">{p}</span>
                ))}
              </span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

function WorkflowBox({ icon, title, sub, color, isOutput = false }: {
  icon: React.ReactNode;
  title: string;
  sub: string;
  color: string;
  isOutput?: boolean;
}) {
  return (
    <div className={`wf-box${isOutput ? " wf-box--output" : ""}`} style={{ "--wf-color": color } as React.CSSProperties}>
      <div className="wf-box__icon">{icon}</div>
      <div className="wf-box__title">{title}</div>
      <div className="wf-box__sub">{sub}</div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="wf-arrow">
      <svg width="24" height="16" viewBox="0 0 24 16">
        <line x1="0" y1="8" x2="18" y2="8" stroke="#d1d5db" strokeWidth="1.5" />
        <polyline points="13,3 19,8 13,13" fill="none" stroke="#d1d5db" strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function DataIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <rect x="3" y="3" width="14" height="3" rx="1" fill="currentColor" opacity="0.7" />
      <rect x="3" y="8.5" width="14" height="3" rx="1" fill="currentColor" opacity="0.5" />
      <rect x="3" y="14" width="9" height="3" rx="1" fill="currentColor" opacity="0.3" />
    </svg>
  );
}

function ModelIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="3" fill="currentColor" />
      <circle cx="3" cy="5" r="2" fill="currentColor" opacity="0.5" />
      <circle cx="17" cy="5" r="2" fill="currentColor" opacity="0.5" />
      <circle cx="3" cy="15" r="2" fill="currentColor" opacity="0.5" />
      <circle cx="17" cy="15" r="2" fill="currentColor" opacity="0.5" />
      <line x1="5" y1="6" x2="8" y2="9" stroke="currentColor" strokeWidth="1" opacity="0.4" />
      <line x1="15" y1="6" x2="12" y2="9" stroke="currentColor" strokeWidth="1" opacity="0.4" />
      <line x1="5" y1="14" x2="8" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.4" />
      <line x1="15" y1="14" x2="12" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.4" />
    </svg>
  );
}

function MetricIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <rect x="3" y="14" width="3" height="3" rx="0.5" fill="currentColor" />
      <rect x="8.5" y="10" width="3" height="7" rx="0.5" fill="currentColor" opacity="0.7" />
      <rect x="14" y="5" width="3" height="12" rx="0.5" fill="currentColor" opacity="0.5" />
    </svg>
  );
}

function ProteinIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="4" cy="8" r="2.5" fill="currentColor" opacity="0.6" />
      <circle cx="10" cy="5" r="2.5" fill="currentColor" opacity="0.8" />
      <circle cx="16" cy="8" r="2.5" fill="currentColor" opacity="0.6" />
      <circle cx="10" cy="14" r="2.5" fill="currentColor" opacity="0.4" />
      <line x1="6" y1="7.5" x2="8" y2="5.8" stroke="currentColor" strokeWidth="1" opacity="0.4" />
      <line x1="12" y1="5.8" x2="14" y2="7.5" stroke="currentColor" strokeWidth="1" opacity="0.4" />
      <line x1="10" y1="7.5" x2="10" y2="11.5" stroke="currentColor" strokeWidth="1" opacity="0.4" />
    </svg>
  );
}

function SimIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <path d="M3 15 Q7 5 10 10 Q13 15 17 5" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.8" />
    </svg>
  );
}

function SpeedIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="11" r="7" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.5" />
      <path d="M10 11 L13 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="10" cy="11" r="1.5" fill="currentColor" />
    </svg>
  );
}
