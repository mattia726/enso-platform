"use client";

import React, { useState } from "react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { CaseExplorer } from "@/components/CaseExplorer";
import { PerformanceTab } from "@/components/PerformanceTab";
import { useTheme } from "@/components/ThemeProvider";

type TabId = "explore" | "performance";

type PageProps = {
  params?: Promise<Record<string, string | string[]>>;
  searchParams?: Promise<Record<string, string | string[]>>;
};

const SCROLL_FLOAT_THRESHOLD = 120;

export default function Home({ params, searchParams }: PageProps) {
  React.use(params ?? Promise.resolve({}));
  React.use(searchParams ?? Promise.resolve({}));
  const [tab, setTab] = useState<TabId | null>(null);
  const [scrollRequest, setScrollRequest] = useState<{ target: TabId; id: number } | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const { theme } = useTheme();
  const logoSrc = theme === "dark" ? "/enso-logo-dark.png" : "/enso-logo.png";

  const handleNavClick = (target: TabId) => {
    setTab(target);
    setScrollRequest((request) => ({
      target,
      id: (request?.id ?? 0) + 1,
    }));
  };

  const isFloating = scrollTop > SCROLL_FLOAT_THRESHOLD;

  return (
    <div className="h-screen max-h-screen overflow-hidden flex flex-col bg-[var(--bg)] text-[var(--text)]">
      <header
        className={`z-20 transition-all duration-500 ease-in-out border-[var(--border)] bg-[var(--bg)]/95 backdrop-blur ${
          isFloating
            ? "fixed top-3 right-4 left-4 md:left-auto md:right-6 md:max-w-sm border rounded-2xl shadow-lg py-2 px-4"
            : "sticky top-0 border-b py-4 px-6 md:px-12 w-full"
        }`}
      >
        <div className={`flex items-center w-full ${isFloating ? "justify-end gap-4" : "justify-between"}`}>
          {!isFloating && (
            <a
              href="https://ensohealth.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={logoSrc} alt="Enso Biosciences" className="h-11 w-auto object-contain" />
              <span className="font-semibold text-xl tracking-tight text-[#007ba7] dark:text-[var(--text)]">Enso Biosciences</span>
            </a>
          )}
          <div className="flex items-center gap-2 md:gap-3">
            <nav className="flex gap-2">
              <button
                type="button"
                onClick={() => handleNavClick("explore")}
                className={`px-4 py-3 rounded-xl text-base font-medium transition-colors ${
                  tab === "explore"
                    ? "bg-[var(--surface)] text-[var(--text)]"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                } ${isFloating ? "py-2 px-3 text-sm" : ""}`}
              >
                Case Explorer
              </button>
              <button
                type="button"
                onClick={() => handleNavClick("performance")}
                className={`px-4 py-3 rounded-xl text-base font-medium transition-colors ${
                  tab === "performance"
                    ? "bg-[var(--surface)] text-[var(--text)]"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                } ${isFloating ? "py-2 px-3 text-sm" : ""}`}
              >
                Performance
              </button>
            </nav>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="flex-1 min-h-0 overflow-hidden flex flex-col w-full">
        <CaseExplorer
          onScroll={setScrollTop}
          onActiveViewChange={setTab}
          scrollRequest={scrollRequest}
          performanceSlot={<PerformanceTab />}
        />
      </main>
    </div>
  );
}
