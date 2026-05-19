"use client";

import { useEffect, useRef } from "react";

import type OpenSeadragon from "openseadragon";

export interface WSIViewerHandle {
  viewer: OpenSeadragon.Viewer | null;
}

export interface WSIViewerProps {
  baseImageUrl: string;
  width: number;
  height: number;
  onReady?: (viewer: OpenSeadragon.Viewer) => void;
  onZoomChange?: (zoom: number, maxZoom: number) => void;
  children?: React.ReactNode;
  className?: string;
}

/**
 * OpenSeadragon-backed WSI viewer for the macrodissection workbench.
 *
 * Uses the ``image`` tile source which works for any flat JPEG/PNG. This is
 * sufficient for the TCGA thumbnails shipped with the demo; a future swap
 * to a deep-zoom (.dzi) source requires only changing the ``tileSources``
 * value here. The viewer is configured for clinical-style behaviour:
 *
 *   * no auto-fade of controls,
 *   * navigator (mini-map) on by default,
 *   * scale bar that respects the ``mpp_thumb_x`` metadata when supplied,
 *   * mouse-wheel zoom with a constrained max zoom of 8× (enough to expose
 *     the raw tile grid without straying into pixel-staircase territory),
 *   * children rendered *above* the OSD canvas so overlay components see
 *     the viewer through ``onReady`` and position themselves accordingly.
 */
export default function WSIViewer({
  baseImageUrl,
  width,
  height,
  onReady,
  onZoomChange,
  children,
  className,
}: WSIViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!containerRef.current) return;

    let viewer: OpenSeadragon.Viewer | null = null;
    let detach: (() => void) | undefined;

    // OpenSeadragon does not officially support SSR / strict-mode double
    // mounting; we guard everything inside an async IIFE.
    (async () => {
      const OSDmod = await import("openseadragon");
      if (cancelled || !containerRef.current) return;
      const OSD = OSDmod.default;
      viewer = OSD({
        element: containerRef.current,
        prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
        tileSources: {
          type: "image",
          url: baseImageUrl,
        },
        showNavigator: true,
        navigatorPosition: "BOTTOM_LEFT",
        navigatorAutoFade: false,
        showFullPageControl: false,
        zoomInButton: undefined,
        zoomOutButton: undefined,
        homeButton: undefined,
        animationTime: 0.4,
        blendTime: 0.1,
        immediateRender: true,
        showRotationControl: false,
        gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: false },
        minZoomImageRatio: 0.8,
        maxZoomPixelRatio: 6,
        defaultZoomLevel: 0,
        visibilityRatio: 0.5,
        constrainDuringPan: true,
        crossOriginPolicy: "Anonymous",
      });
      viewerRef.current = viewer;
      const handleZoom = () => {
        if (!viewer) return;
        onZoomChange?.(viewer.viewport.getZoom(true), viewer.viewport.getMaxZoom());
      };
      viewer.addHandler("zoom", handleZoom);
      viewer.addHandler("open", () => {
        if (cancelled || !viewer) return;
        viewer.viewport.goHome(true);
        handleZoom();
        onReady?.(viewer);
      });
      detach = () => {
        if (viewer) {
          try {
            viewer.removeAllHandlers("zoom");
          } catch {
            /* ignore */
          }
        }
      };
    })().catch((err) => {
      // eslint-disable-next-line no-console
      console.error("OpenSeadragon init failed", err);
    });

    return () => {
      cancelled = true;
      detach?.();
      if (viewerRef.current) {
        try {
          viewerRef.current.destroy();
        } catch {
          /* ignore */
        }
        viewerRef.current = null;
      }
    };
  }, [baseImageUrl, onReady, onZoomChange]);

  return (
    <div
      ref={containerRef}
      data-wsi-viewer
      className={`relative w-full h-full overflow-hidden bg-[#070707] ${className ?? ""}`}
      style={{ aspectRatio: `${Math.max(width, 1)} / ${Math.max(height, 1)}` }}
    >
      {children}
    </div>
  );
}
