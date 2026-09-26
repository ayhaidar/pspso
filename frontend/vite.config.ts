import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../pspso/dashboard/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/zrender/") || id.includes("node_modules/tslib/")) return "chart-renderer";
          if (id.includes("node_modules/echarts/")) return "chart-engine";
        }
      }
    }
  },
  server: {
    proxy: {
      "/api": process.env.PSPSO_API_URL ?? "http://127.0.0.1:8000"
    }
  }
});
