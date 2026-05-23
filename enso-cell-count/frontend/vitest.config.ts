import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@": root,
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.spec.ts", "tests/**/*.spec.tsx"],
    exclude: ["tests/visual/**", "node_modules/**"],
    setupFiles: ["tests/setup.ts"],
    globals: true,
    reporters: ["default"],
  },
});
