import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // This dev environment's native FS-event file watching (FSEvents)
    // unreliably misses edits written by external tooling — HMR silently
    // kept serving stale modules with no error. Polling guarantees changes
    // are picked up regardless of how a file was written.
    watch: { usePolling: true, interval: 300 },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8082',
        changeOrigin: true,
      },
    },
  },
})
