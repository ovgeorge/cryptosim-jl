#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHUNK_ROOT="${CHUNK_ROOT:-${ROOT}/artifacts/chunks_ethusdt-1m-full}"
DATA_DIR="${DATA_DIR:-${ROOT}/cryptopool-simulator/download}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT}/reports/ethusdt_full_chunk_summary.jsonl}"
PROJECT_PATH="${PROJECT_PATH:-${ROOT}}"
JOBS="${JOBS:-$(nproc)}"
SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-0}" # 0 means use all chunks

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options] [chunk_id...]

Options:
  --root DIR            Chunk root directory (default: ${CHUNK_ROOT})
  --data-dir DIR        Raw data directory (default: ${DATA_DIR})
  --output PATH         Output JSONL path (default: ${OUTPUT_PATH})
  --project DIR         Julia project path (default: ${PROJECT_PATH})
  --jobs N              Parallel jobs (default: ${JOBS})
  --sample-chunks N     Randomly sample N chunks (default: ${SAMPLE_CHUNKS})
  -h, --help            Show this help message

Environment variables CHUNK_ROOT, DATA_DIR, OUTPUT_PATH, PROJECT_PATH, JOBS, and
SAMPLE_CHUNKS provide the same overrides.
USAGE
}

CHUNK_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root=*)
            CHUNK_ROOT="${1#--root=}"
            ;;
        --root)
            shift
            [[ $# -gt 0 ]] || { echo "--root requires a directory" >&2; exit 1; }
            CHUNK_ROOT="$1"
            ;;
        --data-dir=*)
            DATA_DIR="${1#--data-dir=}"
            ;;
        --data-dir)
            shift
            [[ $# -gt 0 ]] || { echo "--data-dir requires a directory" >&2; exit 1; }
            DATA_DIR="$1"
            ;;
        --output=*)
            OUTPUT_PATH="${1#--output=}"
            ;;
        --output)
            shift
            [[ $# -gt 0 ]] || { echo "--output requires a path" >&2; exit 1; }
            OUTPUT_PATH="$1"
            ;;
        --project=*)
            PROJECT_PATH="${1#--project=}"
            ;;
        --project)
            shift
            [[ $# -gt 0 ]] || { echo "--project requires a path" >&2; exit 1; }
            PROJECT_PATH="$1"
            ;;
        --jobs=*)
            JOBS="${1#--jobs=}"
            ;;
        --jobs)
            shift
            [[ $# -gt 0 ]] || { echo "--jobs requires a value" >&2; exit 1; }
            JOBS="$1"
            ;;
        --sample-chunks=*|--sample=*)
            SAMPLE_CHUNKS="${1#*=}"
            ;;
        --sample-chunks|--sample)
            shift
            [[ $# -gt 0 ]] || { echo "--sample-chunks requires a value" >&2; exit 1; }
            SAMPLE_CHUNKS="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            CHUNK_ARGS+=("$1")
            ;;
    esac
    shift
done

echo "[full-parity] chunk root : ${CHUNK_ROOT}"
echo "[full-parity] data dir   : ${DATA_DIR}"
echo "[full-parity] jobs       : ${JOBS}"
echo "[full-parity] output     : ${OUTPUT_PATH}"
echo "[full-parity] sample size: ${SAMPLE_CHUNKS}"

if [[ ${#CHUNK_ARGS[@]} -gt 0 ]]; then
    CHUNKS=("${CHUNK_ARGS[@]}")
else
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
fi

scripts/run_parity_parallel.sh \
    --root="${CHUNK_ROOT}" \
    --data-dir="${DATA_DIR}" \
    --project="${PROJECT_PATH}" \
    --output="${OUTPUT_PATH}" \
    --jobs="${JOBS}" \
    "${CHUNKS[@]}"

echo "[full-parity] computing quantiles…"
julia --project="${PROJECT_PATH}" "${ROOT}/scripts/parity_quantiles.jl" "${OUTPUT_PATH}"
