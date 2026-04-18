/**
 * BansuriAI-V2 — API Service
 *
 * Single module for all backend communication.
 *
 * Works in two modes:
 *   DEV:  Vite proxy forwards /analyze and /health to localhost:8000.
 *         API_BASE stays empty (same-origin request via proxy).
 *   PROD: FastAPI serves the built frontend at localhost:8000.
 *         API_BASE stays empty (same-origin, no proxy needed).
 *
 * If neither works (e.g., running the built frontend separately),
 * the fallback URL hits the backend directly.
 */

const API_BASE = '';
const FALLBACK_URL = 'http://localhost:8000';

const ANALYZE_TIMEOUT_MS = 60_000;   // 60 seconds — batch file analysis
const CLASSIFY_TIMEOUT_MS = 8_000;   // 8 seconds — real-time chunk
const HEALTH_TIMEOUT_MS = 5_000;     // 5 seconds

/**
 * Upload a .wav file and get the full analysis response.
 *
 * @param {File} file - A .wav File object from the file input.
 * @returns {Promise<Object>} The AnalysisResponse JSON.
 * @throws {Error} With a user-friendly message on failure.
 */
export async function analyzeAudio(file) {
  const formData = new FormData();
  formData.append('file', file);

  // Try the primary URL (same-origin via proxy or static serving),
  // then fall back to the direct backend URL if that fails.
  let response = await _fetchWithFallback('/analyze', {
    method: 'POST',
    body: formData,
  }, ANALYZE_TIMEOUT_MS);

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(detail);
  }

  return response.json();
}

/**
 * Send a Float32Array PCM chunk for real-time single-note classification.
 *
 * @param {Float32Array} pcmFloat32 - Raw audio samples at `sampleRate` Hz.
 * @param {number} sampleRate - Sample rate of the PCM data (default 22050).
 * @returns {Promise<{ predicted_note, confidence, intonation, cents_off, feedback }>}
 * @throws {Error} With a user-friendly message on failure.
 */
export async function classifyAudio(pcmFloat32, sampleRate = 22050) {
  const audio_b64 = _float32ToBase64(pcmFloat32);

  const response = await _fetchWithFallback('/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_b64, sample_rate: sampleRate }),
  }, CLASSIFY_TIMEOUT_MS);

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(detail);
  }

  return response.json();
}

/**
 * Check backend health and model loading status.
 *
 * @returns {Promise<{ status: string, model_loaded: boolean }>}
 * @throws {Error} If the backend is not reachable.
 */
export async function checkHealth() {
  const response = await _fetchWithFallback('/health', {
    method: 'GET',
  }, HEALTH_TIMEOUT_MS);

  if (!response.ok) throw new Error('Health check failed');
  return response.json();
}


// ═══════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════

/**
 * Fetch with AbortController timeout and fallback URL.
 *
 * First tries API_BASE + path (same-origin). If that fails with a
 * network error (not an HTTP error — those are valid responses),
 * retries with FALLBACK_URL + path.
 */
async function _fetchWithFallback(path, options, timeoutMs) {
  // Attempt 1: same-origin (works in both dev proxy and prod static)
  try {
    return await _fetchWithTimeout(`${API_BASE}${path}`, options, timeoutMs);
  } catch (primaryError) {
    // Only fall back on network errors, not HTTP errors
    if (primaryError.name === 'AbortError') {
      throw new Error(
        'Request timed out. The audio file may be too large, ' +
        'or the backend is processing slowly.'
      );
    }

    // If API_BASE is already empty and we're on localhost:8000,
    // no point trying the fallback (it's the same URL).
    if (!API_BASE && window.location.port === '8000') {
      throw new Error(
        'Cannot reach the backend. Make sure the server is running.'
      );
    }

    // Attempt 2: direct backend URL (for when frontend runs standalone)
    try {
      return await _fetchWithTimeout(`${FALLBACK_URL}${path}`, options, timeoutMs);
    } catch {
      throw new Error(
        'Cannot reach the backend server at localhost:8000. ' +
        'Start it with: cd backend && python run.py'
      );
    }
  }
}

/**
 * Fetch with an AbortController timeout.
 */
async function _fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Convert a Float32Array of PCM samples to a base64 string.
 * The bytes are the raw IEEE 754 float32 representation (little-endian).
 */
function _float32ToBase64(float32Array) {
  const bytes = new Uint8Array(float32Array.buffer);
  let binary = '';
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

/**
 * Extract a user-friendly error message from a failed response.
 */
async function _extractErrorDetail(response) {
  let detail = `Server error (${response.status})`;

  // Map common HTTP status codes to friendlier messages
  const messages = {
    400: 'Invalid file. Please upload a .wav audio file.',
    413: 'File too large.',
    422: 'Could not process the audio.',
    500: 'An internal error occurred. Please try again.',
  };

  try {
    const body = await response.json();
    // Backend sends { "detail": "..." } for errors
    if (body.detail) {
      detail = body.detail;
    }
  } catch {
    // Response wasn't JSON — use status-based message
    detail = messages[response.status] || detail;
  }

  return detail;
}
