import { useEffect, useRef, useState } from "react";

const DMOL_JS = "https://cdn.jsdelivr.net/npm/3dmol@latest/build/3Dmol-min.js";

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if ((window as any).$3Dmol) { resolve(); return; }
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`);
    if (existing) {
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", reject);
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load 3Dmol.js"));
    document.head.appendChild(s);
  });
}

function Spinner() { return <div className="spinner" />; }

function themeSurface() {
  return getComputedStyle(document.documentElement).getPropertyValue("--surface").trim() || "white";
}

interface Props {
  height?: number;
  showControls?: boolean;
  url?: string;
}

export function TrajectoryViewer({ height = 300, showControls = true, url }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!containerRef.current) return;
    let cancelled = false;
    setStatus("loading");
    setPlaying(true);

    async function init() {
      try {
        await loadScript(DMOL_JS);
        if (cancelled || !containerRef.current) return;

        const $3Dmol = (window as any).$3Dmol;
        const viewer = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: themeSurface(),
          antialias: true,
          id: "traj-" + Math.random().toString(36).slice(2),
        });
        viewerRef.current = viewer;

        const trajUrl = import.meta.env.BASE_URL + (url ?? "data/trajectories/chignolin.xyz");
        const data = await fetch(trajUrl).then((r) => r.text());
        if (cancelled) return;

        viewer.addModelsAsFrames(data, "xyz");
        viewer.setStyle({}, { stick: { radius: 0.12, colorscheme: "Jmol" } });
        viewer.zoomTo();
        viewer.render();
        viewer.animate({ loop: "forward", reps: 0 });

        if (!cancelled) setStatus("ready");
      } catch (e) {
        if (!cancelled) setStatus("error");
      }
    }

    init();

    return () => {
      cancelled = true;
      if (viewerRef.current) {
        try { viewerRef.current.stopAnimate(); } catch (_) { /* ignore */ }
        viewerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const updateViewerBackground = () => {
      if (viewerRef.current?.setBackgroundColor) {
        viewerRef.current.setBackgroundColor(themeSurface());
        viewerRef.current.render();
      }
    };
    const observer = new MutationObserver(updateViewerBackground);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    updateViewerBackground();
    return () => observer.disconnect();
  }, []);

  function togglePlay() {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (playing) {
      viewer.stopAnimate();
    } else {
      viewer.animate({ loop: "forward", reps: 0 });
    }
    setPlaying((p) => !p);
  }

  return (
    <div>
      <div style={{ position: "relative", width: "100%", height, borderRadius: 6, overflow: "hidden", border: "1px solid var(--border-soft)" }}>
        {status === "loading" && (
          <div style={{
            position: "absolute", inset: 0, display: "flex", alignItems: "center",
            justifyContent: "center", gap: 8, background: "var(--surface-subtle)",
            fontSize: 12, color: "var(--text-faint)", zIndex: 1,
          }}>
            <Spinner /> Loading…
          </div>
        )}
        {status === "error" && (
          <div style={{
            position: "absolute", inset: 0, display: "flex", alignItems: "center",
            justifyContent: "center", background: "var(--danger-soft)",
            fontSize: 11, color: "var(--danger)", zIndex: 1,
          }}>
            Failed to load
          </div>
        )}
        <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
      </div>

      {showControls && status === "ready" && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
          <button
            onClick={togglePlay}
            style={{
              padding: "4px 14px", fontSize: 11, fontFamily: "inherit",
              border: "1.5px solid var(--border)", borderRadius: 999,
              background: "transparent", color: "var(--text)", cursor: "pointer",
            }}
          >
            {playing ? "⏸ Pause" : "▶ Play"}
          </button>
          <span style={{ fontSize: 11, color: "var(--text-faint)" }}>
            chignolin · 138 atoms · 22 frames · UMA-s-1.2
          </span>
        </div>
      )}
    </div>
  );
}
