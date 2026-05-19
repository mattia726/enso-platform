"use client";

import { useState, useEffect } from "react";

const PHRASE_1 = { text: "Trained on 15,000+ slides." };
const PHRASE_2 = { text: "Across 32 cancer types." };
const PHRASE_3 = { text: "2× more accurate than human pathologists." };

// Left-to-right gradient: more orange on the left, then purple
const LINE_1_GRADIENT =
  "linear-gradient(to right, #ea580c 0%, #ea580c 35%, #c026d3 70%, #a855f7 100%)";
const LINE_2_GRADIENT =
  "linear-gradient(to right, #34d399 0%, #10b981 60%, #6ee7b7 100%)";

const MS_PER_CHAR = 45;
const PAUSE_BETWEEN_LINES_MS = 400;

// Dot uses the hue between line-1 gradient (orange → purple); #d84361 at center
const MID_DOT_GRADIENT =
  "linear-gradient(to right, #ea580c 0%, #d84361 50%, #a855f7 100%)";

function MidDot() {
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full mx-3 shrink-0 align-middle"
      style={{ background: MID_DOT_GRADIENT }}
      aria-hidden
    />
  );
}

export function HeroTypewriter() {
  const [len1, setLen1] = useState(0);
  const [len2, setLen2] = useState(0);
  const [len3, setLen3] = useState(0);

  useEffect(() => {
    const n1 = PHRASE_1.text.length;
    const n2 = PHRASE_2.text.length;
    const n3 = PHRASE_3.text.length;
    const pauseTicks = Math.max(1, Math.round(PAUSE_BETWEEN_LINES_MS / MS_PER_CHAR));
    let phase: "1" | "pause1" | "2" | "pause2" | "3" = "1";
    let charIndex = 0;
    let pauseLeft = 0;

    const tick = () => {
      if (phase === "1") {
        if (charIndex < n1) {
          charIndex += 1;
          setLen1(charIndex);
          return;
        }
        phase = "pause1";
        pauseLeft = pauseTicks;
      }
      if (phase === "pause1") {
        if (pauseLeft > 0) {
          pauseLeft -= 1;
          return;
        }
        phase = "2";
        charIndex = 0;
      }
      if (phase === "2") {
        if (charIndex < n2) {
          charIndex += 1;
          setLen2(charIndex);
          return;
        }
        phase = "pause2";
        pauseLeft = pauseTicks;
      }
      if (phase === "pause2") {
        if (pauseLeft > 0) {
          pauseLeft -= 1;
          return;
        }
        phase = "3";
        charIndex = 0;
      }
      if (phase === "3") {
        if (charIndex < n3) {
          charIndex += 1;
          setLen3(charIndex);
        }
      }
    };

    const id = setInterval(tick, MS_PER_CHAR);
    return () => clearInterval(id);
  }, []);

  const visible1 = PHRASE_1.text.slice(0, len1);
  const visible2 = PHRASE_2.text.slice(0, len2);
  const visible3 = PHRASE_3.text.slice(0, len3);

  return (
    <div className="text-center max-w-3xl mx-auto space-y-1">
      {/* Line 1: gradient left→right, phrases closer together */}
      <p
        className="text-2xl md:text-3xl font-bold leading-tight min-h-[1.4em] flex flex-wrap items-center justify-center gap-x-0 gap-y-1"
        style={{
          background: LINE_1_GRADIENT,
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
          color: "transparent",
        }}
      >
        <span>{visible1}</span>
        {len1 < PHRASE_1.text.length && <span className="animate-pulse opacity-90">|</span>}
        {len1 >= PHRASE_1.text.length && <MidDot />}
        <span>{visible2}</span>
        {len1 >= PHRASE_1.text.length && len2 < PHRASE_2.text.length && <span className="animate-pulse opacity-90">|</span>}
      </p>
      {/* Line 2: gradient green, slides in nicely */}
      {len3 > 0 && (
        <p
          className="text-2xl md:text-3xl font-bold leading-tight min-h-[1.4em] animate-slide-in"
          style={{
            background: LINE_2_GRADIENT,
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
          }}
        >
          <span className="italic border-b-2" style={{ borderBottomColor: "rgba(234,88,12,0.9)" }}>
            2× more accurate
          </span>
          {visible3.slice("2× more accurate".length)}
          {len3 < PHRASE_3.text.length && <span className="animate-pulse opacity-90">|</span>}
        </p>
      )}
    </div>
  );
}
