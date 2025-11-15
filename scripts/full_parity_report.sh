#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHUNK_ROOT="${CHUNK_ROOT:-${ROOT}/artifacts/chunks_ethusdt-1m-full}"
DATA_DIR="${DATA_DIR:-${ROOT}/cryptopool-simulator/download}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT}/reports/ethusdt_full_chunk_summary.jsonl}"
PROJECT_PATH="${ROOT}"
JOBS="${JOBS:-$(nproc)}"
SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-0}" # 0 means use all chunks

echo "[full-parity] chunk root : ${CHUNK_ROOT}"
echo "[full-parity] data dir   : ${DATA_DIR}"
echo "[full-parity] jobs       : ${JOBS}"
echo "[full-parity] output     : ${OUTPUT_PATH}"
echo "[full-parity] sample size: ${SAMPLE_CHUNKS}"

mapfile -t ALL_CHUNKS < <(find "${CHUNK_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'chunk*' -printf '%f\n' | sort)
if [[ ${#ALL_CHUNKS[@]} -eq 0 ]]; then
    echo "No chunk directories under ${CHUNK_ROOT}" >&2
    exit 1
fi

CHUNKS=("${ALL_CHUNKS[@]}")
if [[ ${SAMPLE_CHUNKS} -gt 0 && ${SAMPLE_CHUNKS} -lt ${#ALL_CHUNKS[@]} ]]; then
    shuf_chunks=($(printf "%s\n" "${ALL_CHUNKS[@]}" | shuf))
    CHUNKS=("${shuf_chunks[@]:0:${SAMPLE_CHUNKS}}")
    echo "[full-parity] sampled ${#CHUNKS[@]} chunks"
fi

scripts/run_parity_parallel.sh \
    --root="${CHUNK_ROOT}" \
    --data-dir="${DATA_DIR}" \
    --project="${PROJECT_PATH}" \
    --output="${OUTPUT_PATH}" \
    --jobs="${JOBS}" \
    "${CHUNKS[@]}"

echo "[full-parity] computing quantiles…"
julia --project="${PROJECT_PATH}" "${ROOT}/scripts/parity_quantiles.jl"
