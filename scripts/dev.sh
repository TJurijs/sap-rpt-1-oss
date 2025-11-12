#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BACKEND_DIR="${PROJECT_ROOT}/playground/backend"
FRONTEND_DIR="${PROJECT_ROOT}/playground/frontend"
SAP_RPT_PACKAGE_DIR="${PROJECT_ROOT}/sap-rpt-1-oss"
SAP_RPT_ARCHIVE_URL="${SAP_RPT_SOURCE_URL:-https://codeload.github.com/SAP-samples/sap-rpt-1-oss/tar.gz/refs/heads/main}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  echo "Loading environment variables from .env"
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

ensure_sap_rpt_package() {
  SAP_RPT_INSTALL_ARGS=("-e" "../../sap-rpt-1-oss")

  if [[ -d "${SAP_RPT_PACKAGE_DIR}" ]]; then
    echo "Using existing sap-rpt-1-oss sources at ${SAP_RPT_PACKAGE_DIR}"
    return
  fi

  echo "sap-rpt-1-oss sources not found. Downloading snapshot..."

  if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    echo "Error: both curl and tar are required to download sap-rpt-1-oss. Install the package manually and rerun the script." >&2
    exit 1
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"

  if ! curl -fSL "${SAP_RPT_ARCHIVE_URL}" \
    | tar -xz -C "${tmp_dir}"; then
    echo "Error: failed to download sap-rpt-1-oss sources from ${SAP_RPT_ARCHIVE_URL}." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  local extracted_dir
  extracted_dir="$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"

  if [[ -z "${extracted_dir}" || ! -d "${extracted_dir}" ]]; then
    echo "Error: sap-rpt-1-oss archive did not unpack correctly." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  mv "${extracted_dir}" "${SAP_RPT_PACKAGE_DIR}"
  rm -rf "${tmp_dir}"
}

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && ps -p "${BACKEND_PID}" > /dev/null 2>&1; then
    echo "Stopping backend (PID ${BACKEND_PID})"
    kill "${BACKEND_PID}" || true
  fi
}

trap cleanup EXIT INT TERM

wait_for_backend() {
  local attempts=0
  local max_attempts=60

  echo "Waiting for backend to become ready..."
  until curl -sf "http://127.0.0.1:8000/api/health" > /dev/null 2>&1; do
    if ! ps -p "${BACKEND_PID}" > /dev/null 2>&1; then
      echo "Backend process exited unexpectedly. Check logs above for details."
      exit 1
    fi
    attempts=$((attempts + 1))
    if [[ ${attempts} -ge ${max_attempts} ]]; then
      echo "Timed out waiting for backend on http://127.0.0.1:8000/api/health"
      exit 1
    fi
    sleep 2
  done
  echo "Backend is ready."
}

ensure_sap_rpt_package

echo "Starting backend (uvicorn)..."
(
  cd "${BACKEND_DIR}"
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
  if [[ ! -d ".venv" ]]; then
    python3.11 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install "${SAP_RPT_INSTALL_ARGS[@]}"
  fi
  source .venv/bin/activate
  uvicorn playground.backend.main:app --host 0.0.0.0 --port 8000
) &

BACKEND_PID=$!
echo "Backend started with PID ${BACKEND_PID}"

wait_for_backend

echo "Starting frontend (npm run dev)..."
cd "${FRONTEND_DIR}"
if [[ ! -d "node_modules" ]]; then
  npm install
fi
npm run dev -- --host 0.0.0.0


