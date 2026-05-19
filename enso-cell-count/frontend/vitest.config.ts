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
    include: ["tests/**/*.spec.ts", "tests/**/*.test.ts"],
    exclude: ["tests/visual/**", "node_modules/**"],
    globals: true,
    reporters: ["default"],
  },
});
