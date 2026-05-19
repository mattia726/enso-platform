"use client";

import { useState, useEffect, useMemo } from "react";

export type ScatterData = {
  genomic_purity: number[];
  enso_mil: number[];
  pathologist_ptn: number[];
};

const PAD = 40;
const DOMAIN_MIN = -0.05;
const DOMAIN_MAX = 1.05;

function toSvg(x: number, y: number, width: number, height: number) {
  const w = width - 2 * PAD;
  const h = height - 2 * PAD;
  const sx = PAD + ((x - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * w;
  const sy = height - PAD - ((y - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * h;
  return { x: sx, y: sy };
}

type SingleScatterProps = {
  data: ScatterData;
  series: "enso_mil" | "pathologist_ptn";
  title: string;
  width: number;
  height: number;
  pointRadius?: number;
  onHover?: (index: number | null) => void;
  hoveredIndex: number | null;
  pointColor: string;
  diagonalColor: string;
};

function SingleScatter({
  data,
  series,
  title,
  width,
  height,
  pointRadius = 3,
  onHover,
  hoveredIndex,
  pointColor,
  diagonalColor,
}: SingleScatterProps) {
  const genomic = data.genomic_purity;
  const yArr = series === "enso_mil" ? data.enso_mil : data.pathologist_ptn;

  const points = useMemo(() => {
    return genomic.map((x, i) => ({
      ...toSvg(x, yArr[i], width, height),
      index: i,
      xVal: x,
      yVal: yArr[i],
    }));
  }, [genomic, yArr, width, height]);

  const diagStart = toSvg(DOMAIN_MIN, DOMAIN_MIN, width, height);
  const diagEnd = toSvg(DOMAIN_MAX, DOMAIN_MAX, width, height);

  return (
    <div className="relative flex flex-col items-center">
      <p className="text-xs font-medium mb-1" style={{ color: "var(--text)" }}>
        {title}
      </p>
      <svg
        width={width}
        height={height}
        className="overflow-visible"
        onMouseLeave={() => onHover?.(null)}
      >
        <line
          x1={diagStart.x}
          y1={diagStart.y}
          x2={diagEnd.x}
          y2={diagEnd.y}
          stroke={diagonalColor}
          strokeWidth={1}
          strokeDasharray="4,2"
          opacity={0.6}
        />
        {points.map((p) => (
          <circle
            key={p.index}
            cx={p.x}
            cy={p.y}
            r={pointRadius}
            fill={pointColor}
            opacity={hoveredIndex === p.index ? 0.95 : 0.35}
            stroke={hoveredIndex === p.index ? "var(--text)" : "none"}
            strokeWidth={1}
            onMouseEnter={() => onHover?.(p.index)}
          />
        ))}
      </svg>
    </div>
  );
}

type ScatterChartProps = {
  data: ScatterData | null;
  className?: string;
  variant?: "default" | "background";
};

export function ScatterChart({ data, className = "", variant = "default" }: ScatterChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const chartSize = 280;
  const isBackground = variant === "background";

  const pointColor = "var(--accent)";
  const diagonalColor = "var(--border)";

  if (!data || !data.genomic_purity?.length) return null;

  const n = data.genomic_purity.length;
  const hoveredPoint =
    hoveredIndex != null && hoveredIndex >= 0 && hoveredIndex < n
      ? {
          genomic: data.genomic_purity[hoveredIndex],
          enso: data.enso_mil[hoveredIndex],
          ptn: data.pathologist_ptn[hoveredIndex],
        }
      : null;

  const content = (
    <div
      className={variant === "default" ? `relative ${className}` : className}
      style={
        isBackground
          ? {
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
              opacity: 0.08,
              zIndex: 0,
            }
          : undefined
      }
    >
      <div
        className={variant === "default" ? "flex flex-wrap justify-center gap-8" : ""}
        style={isBackground ? { width: "100%", height: "100%" } : undefined}
      >
        <SingleScatter
          data={data}
          series="enso_mil"
          title="Enso MIL"
          width={chartSize}
          height={chartSize}
          onHover={isBackground ? undefined : setHoveredIndex}
          hoveredIndex={hoveredIndex}
          pointColor={pointColor}
          diagonalColor={diagonalColor}
        />
        <SingleScatter
          data={data}
          series="pathologist_ptn"
          title="Pathologist PTN"
          width={chartSize}
          height={chartSize}
          onHover={isBackground ? undefined : setHoveredIndex}
          hoveredIndex={hoveredIndex}
          pointColor="var(--warning)"
          diagonalColor={diagonalColor}
        />
      </div>
      {!isBackground && hoveredPoint && (
        <div
          className="absolute bottom-2 left-1/2 -translate-x-1/2 px-3 py-2 rounded-lg border text-xs shadow-lg z-10"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
            color: "var(--text)",
          }}
        >
          Genomic: {hoveredPoint.genomic.toFixed(3)} · Enso: {hoveredPoint.enso.toFixed(3)} · PTN:{" "}
          {hoveredPoint.ptn.toFixed(3)}
        </div>
      )}
    </div>
  );

  return content;
}

type ScatterSectionProps = {
  variant?: "default" | "background";
};

export function ScatterSection({ variant = "default" }: ScatterSectionProps) {
  const [data, setData] = useState<ScatterData | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch("/data/scatter_data.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("404"))))
      .then(setData)
      .catch(() => setFailed(true));
  }, []);

  if (failed || !data) return null;
  return <ScatterChart data={data} variant={variant} />;
}
