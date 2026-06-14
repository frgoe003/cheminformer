import type { GpuId, GpuInstance } from "../data";
import { GPU_INSTANCES } from "../data";

interface Props {
  selected: GpuId;
  onChange: (id: GpuId) => void;
}

function SpecChip({ label, value }: { label: string; value: string }) {
  return (
    <span className="gpu-spec">
      <span className="gpu-spec__label">{label}</span>
      <span className="gpu-spec__value">{value}</span>
    </span>
  );
}

function GpuCard({ inst, active, onClick }: { inst: GpuInstance; active: boolean; onClick: () => void }) {
  return (
    <button
      className={`gpu-card${active ? " gpu-card--active" : ""}`}
      onClick={onClick}
    >
      <div className="gpu-card__header">
        <span className="gpu-card__instance">{inst.instanceType}</span>
        {active && <span className="gpu-card__badge">selected</span>}
      </div>
      <div className="gpu-card__gpu">{inst.gpu}</div>
      <div className="gpu-card__specs">
        <SpecChip label="VRAM" value={inst.vram} />
        <SpecChip label="vCPUs" value={String(inst.vcpus)} />
        <SpecChip label="RAM" value={inst.ram} />
      </div>
    </button>
  );
}

export function GpuSelector({ selected, onChange }: Props) {
  return (
    <div className="gpu-selector">
      {(Object.values(GPU_INSTANCES) as GpuInstance[]).map((inst) => (
        <GpuCard
          key={inst.id}
          inst={inst}
          active={inst.id === selected}
          onClick={() => onChange(inst.id)}
        />
      ))}
    </div>
  );
}
