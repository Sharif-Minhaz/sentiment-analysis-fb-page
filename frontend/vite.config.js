import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],

  server: {
    port: 5173,
    // In development the Vue dev server and Flask are separate origins. Rather
    // than enable CORS on Flask, proxy /api through Vite -- the browser only
    // ever talks to one origin, so there is nothing to configure on the server
    // and dev behaves exactly like production.
    proxy: {
      '/api': { target: 'http://127.0.0.1:5000', changeOrigin: true },
    },
  },

  build: {
    // Build straight into Flask's static folder, so `npm run build` is all it
    // takes to ship: Flask then serves the compiled app at http://127.0.0.1:5000
    // with no second process.
    outDir: '../backend/static',
    emptyOutDir: true,
  },
})
