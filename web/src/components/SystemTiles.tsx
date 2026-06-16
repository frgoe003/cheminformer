import { PERF_SYSTEMS } from "../data";

export function SystemTiles() {
  return (
    <div className="system-tiles">
      {PERF_SYSTEMS.map(({ system, n_atoms, trajectoryGif }) => (
        <div key={system} className="system-tile">
          <div className="system-tile__viewer">
            {trajectoryGif ? (
              <img
                src={import.meta.env.BASE_URL + trajectoryGif}
                alt={`${system} MD trajectory`}
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
                loading="lazy"
              />
            ) : (
              <AtomDot n={n_atoms} />
            )}
          </div>
          <div className="system-tile__name">{system}</div>
          <div className="system-tile__count">{n_atoms.toLocaleString()} atoms</div>
        </div>
      ))}
    </div>
  );
}

function AtomDot({ n }: { n: number }) {
  const minLog = Math.log10(22);
  const maxLog = Math.log10(99999);
  const t = (Math.log10(n) - minLog) / (maxLog - minLog);
  const r = Math.round(6 + t * 22);
  return (
    <svg width="48" height="48" viewBox="0 0 48 48">
      <circle cx="24" cy="24" r={r} fill="var(--brand-softer)" />
      <circle cx="24" cy="24" r={Math.max(3, r - 5)} fill="var(--brand-border)" />
    </svg>
  );
}
