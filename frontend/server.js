/*
 * NetraChat frontend server (Node.js + Express).
 *
 * It does two jobs:
 *   1. Serves the web page (the files in public/) to your browser.
 *   2. Forwards every request that starts with /api to the Python backend
 *      (api.py), which runs the AI models, and passes the answer back.
 *
 * Run:   npm start            then open http://localhost:3000
 *
 * Settings (environment variables):
 *   PORT         port for this server            (default 3000)
 *   BACKEND_URL  where the Python API is running (default http://127.0.0.1:8000)
 */

import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function createServer({ backendUrl = "http://127.0.0.1:8000" } = {}) {
  const app = express();

  // Forward /api/... to Python. Uploads stream straight through.
  app.use(
    "/api",
    createProxyMiddleware({
      target: backendUrl,
      changeOrigin: true,
      pathRewrite: (p) => "/api" + p, // Express strips "/api"; put it back
      proxyTimeout: 120_000,
      on: {
        error: (err, req, res) => {
          // The Python backend is not running (or crashed).
          if (!res.headersSent) {
            res.writeHead(502, { "Content-Type": "application/json" });
          }
          res.end(
            JSON.stringify({
              detail:
                "Cannot reach the NetraChat backend. Start it with: python api.py",
            }),
          );
        },
      },
    }),
  );

  // The web page itself.
  app.use(express.static(path.join(__dirname, "public")));

  return app;
}

// Start the server only when this file is run directly (not in tests).
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT) || 3000;
  const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";
  createServer({ backendUrl }).listen(port, () => {
    console.log(`NetraChat frontend:  http://localhost:${port}`);
    console.log(`Forwarding /api to:  ${backendUrl}`);
  });
}
