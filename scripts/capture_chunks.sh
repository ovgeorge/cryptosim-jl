#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="${SIM_DIR:-${ROOT_DIR}/cryptopool-simulator}"
if [ ! -d "${SIM_DIR}" ]; then
  echo "error: SIM_DIR '${SIM_DIR}' not found" >&2
  exit 1
fi
SIM_DIR="$(cd "${SIM_DIR}" && pwd)"
DEFAULT_CONFIG="${SIM_DIR}/single-run.json"
CONFIG_PATH="$(realpath "${1:-$DEFAULT_CONFIG}")"
OUTPUT_DIR="${2:-$ROOT_DIR/test/fixtures/chunks}"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
CHUNKS=("${@:3}")
if [ ${#CHUNKS[@]} -eq 0 ]; then
  CHUNKS=(00972 00980 00984 01071)
fi

mkdir -p "${OUTPUT_DIR}"

pushd "${SIM_DIR}" >/dev/null
make >/dev/null
CPP_COMMIT="$(git rev-parse HEAD || echo "unknown")"
SIM_SHA="$(sha256sum simu | awk '{print $1}')"
popd >/dev/null

for CHUNK in "${CHUNKS[@]}"; do
  CHUNK_DIR="${OUTPUT_DIR}/${CHUNK}"
  mkdir -p "${CHUNK_DIR}"
  LOG_FILE="${CHUNK_DIR}/cpp_stdout.log"
  RESULT_FILE="${CHUNK_DIR}/results.json"
  CONFIG_BASENAME="$(basename "${CONFIG_PATH}")"
  cp "${CONFIG_PATH}" "${CHUNK_DIR}/${CONFIG_BASENAME}"

  pushd "${SIM_DIR}" >/dev/null
  CMD=(./simu "trim${CHUNK}" "${CONFIG_PATH}" "${RESULT_FILE}")
  echo "Running ${CMD[*]} (CPP_TRADE_DEBUG=1 CPP_KEEP_TMP=1)"
  CPP_TRADE_DEBUG=1 CPP_KEEP_TMP=1 "${CMD[@]}" > "${LOG_FILE}"
  TMP_FILE=$(grep -o "_tmp\.[0-9]\+" "${LOG_FILE}" | tail -n1 || true)
  if [[ -n "${TMP_FILE}" && -f "${TMP_FILE}" ]]; then
    cp "${TMP_FILE}" "${CHUNK_DIR}/trades.bin"
    rm -f "${TMP_FILE}"
  else
    echo "warning: temp file not found for chunk ${CHUNK}"
  fi
  popd >/dev/null

  grep '^CPPDBG ' "${LOG_FILE}" | sed 's/^CPPDBG //' > "${CHUNK_DIR}/cpp_log.jsonl" || true

  DATA_SOURCES=$(jq -c '.datafile' "${CONFIG_PATH}")
  cat > "${CHUNK_DIR}/metadata.json" <<JSON
{
  "chunk": "${CHUNK}",
  "trim_flag": "trim${CHUNK}",
  "config": "${CONFIG_BASENAME}",
  "data_sources": ${DATA_SOURCES},
  "cpp_commit": "${CPP_COMMIT}",
  "simu_sha256": "${SIM_SHA}",
  "timestamp": "$(date --iso-8601=seconds)",
  "command": "./simu trim${CHUNK} ${CONFIG_BASENAME} results.json"
}
JSON
done
