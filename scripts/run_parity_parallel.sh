#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CHUNK_ROOT="$ROOT/test/fixtures/chunks"
DEFAULT_DATA_DIR="$ROOT/cryptopool-simulator/download"
PROJECT_PATH="$ROOT"
SUMMARY_SCRIPT="$ROOT/scripts/chunk_summary.jl"
OUTPUT_PATH=""
JOBS=0
EXTRA_ARGS=()
CHUNKS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root=*)
            DEFAULT_CHUNK_ROOT="${1#--root=}"
            ;;
        --data-dir=*)
            DEFAULT_DATA_DIR="${1#--data-dir=}"
            ;;
        --project=*)
            PROJECT_PATH="${1#--project=}"
            ;;
        --script=*)
            SUMMARY_SCRIPT="${1#--script=}"
            ;;
        --jobs=*)
            JOBS="${1#--jobs=}"
            ;;
        --output=*)
            OUTPUT_PATH="${1#--output=}"
            ;;
        --output)
            shift
            [[ $# -gt 0 ]] || { echo "--output requires a path" >&2; exit 1; }
            OUTPUT_PATH="$1"
            ;;
        --*)
            EXTRA_ARGS+=("$1")
            ;;
        *)
            CHUNKS+=("$1")
            ;;
    esac
    shift
done

if [[ ${#CHUNKS[@]} -eq 0 ]]; then
    while IFS= read -r entry; do
        [[ $entry == chunk* ]] || continue
        CHUNKS+=("$entry")
    done < <(find "$DEFAULT_CHUNK_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

if [[ ${#CHUNKS[@]} -eq 0 ]]; then
    echo "No chunk directories found under $DEFAULT_CHUNK_ROOT" >&2
    exit 1
fi

CMD=(julia "--project=$PROJECT_PATH" "$SUMMARY_SCRIPT" "--root=$DEFAULT_CHUNK_ROOT" "--data-dir=$DEFAULT_DATA_DIR")
CMD+=("${EXTRA_ARGS[@]}")
printf -v CMD_STR '%q ' "${CMD[@]}"
export CMD_STR

run_chunk() {
    local chunk="$1"
    eval "$CMD_STR \"\$chunk\""
}
export -f run_chunk

if [[ -n "$OUTPUT_PATH" ]]; then
    parallel --jobs "$JOBS" --tag --line-buffer run_chunk ::: "${CHUNKS[@]}" | tee "$OUTPUT_PATH"
else
    parallel --jobs "$JOBS" --tag --line-buffer run_chunk ::: "${CHUNKS[@]}"
fi
