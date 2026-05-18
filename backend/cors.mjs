/**
 * backend/cors.mjs
 *
 * Centralised CORS configuration for the GridDB REST backend.
 *
 * Environment variables
 * ─────────────────────
 *  CORS_ORIGINS   – comma-separated list of allowed origins.
 *                   Set to "*" to allow every origin (development only).
 *                   Defaults to the origins listed in DEVELOPMENT_ORIGINS below.
 *
 *  CORS_MAX_AGE   – pre-flight cache duration in seconds (default 600 = 10 min)
 *
 * Examples
 *   CORS_ORIGINS=https://app.example.com,https://staging.example.com node server.mjs
 *   CORS_ORIGINS=* node server.mjs   # open – dev only
 */

// Origins permitted when CORS_ORIGINS is not set
const DEVELOPMENT_ORIGINS = [
  "http://localhost:3000",
  "http://localhost:5173",
  "http://localhost:8080",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:5173",
  "http://127.0.0.1:8080",
];

function parseAllowedOrigins() {
  const raw = process.env.CORS_ORIGINS;
  if (!raw) return DEVELOPMENT_ORIGINS;
  if (raw.trim() === "*") return "*";
  return raw.split(",").map((o) => o.trim()).filter(Boolean);
}

const allowedOrigins = parseAllowedOrigins();

/**
 * cors() options object – import this and pass directly to `cors(corsOptions)`.
 *
 * @type {import('cors').CorsOptions}
 */
export const corsOptions = {
  // ── Origin check ──────────────────────────────────────────────────────────
  origin(requestOrigin, callback) {
    // Same-origin requests (Postman, curl, server-to-server) have no Origin header
    if (!requestOrigin) return callback(null, true);

    if (allowedOrigins === "*") return callback(null, true);

    if (allowedOrigins.includes(requestOrigin)) {
      return callback(null, true);
    }

    callback(
      Object.assign(new Error(`CORS: origin "${requestOrigin}" is not allowed`), {
        status: 403,
      }),
    );
  },

  // ── Allowed HTTP methods ───────────────────────────────────────────────────
  methods: ["GET", "POST", "DELETE", "OPTIONS"],

  // ── Allowed request headers ────────────────────────────────────────────────
  allowedHeaders: [
    "Content-Type",
    "Authorization",
    "X-Requested-With",
    "Accept",
  ],

  // ── Exposed response headers (readable by browser JS) ─────────────────────
  exposedHeaders: ["X-Query-Time", "X-Row-Count"],

  // ── Credentials (cookies / Authorization header) ──────────────────────────
  // Keep false unless you need session cookies across origins. Setting this to
  // true while origin is "*" is rejected by browsers anyway.
  credentials: false,

  // ── Pre-flight cache ───────────────────────────────────────────────────────
  maxAge: Number(process.env.CORS_MAX_AGE) || 600,
};
