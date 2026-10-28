#!/bin/bash
# Fuzz every stochastic function with edge-case arguments against a build of the extension.
# Usage: test/fuzz/run.sh [build directory (default build/release)] [output directory (default build/fuzz)]
# Environment: SHARDS (parallel CLI processes, default: CPU count), STALL (seconds without output before a
# statement counts as hung, default 20).
set -eu
cd "$(dirname "$0")/../.."
BUILD=${1:-build/release}
OUT=${2:-build/fuzz}
SHARDS=${SHARDS:-$(getconf _NPROCESSORS_ONLN)}
STALL=${STALL:-20}
FUZZ=test/fuzz
EXTENSION=$BUILD/extension/stochastic/stochastic.duckdb_extension
rm -rf "$OUT" && mkdir -p "$OUT"

$BUILD/duckdb -unsigned -json -c "LOAD '$EXTENSION';
SELECT DISTINCT function_name, function_type, parameters, parameter_types, examples FROM duckdb_functions()
WHERE function_name LIKE 'dist\_%' ESCAPE '\' OR function_name IN ('welch_t_test', 'student_t_test',
  'one_sample_t_test', 'chi_squared_test') ORDER BY ALL" > "$OUT/functions.json"
python3 $FUZZ/fuzz.py gen "$OUT/functions.json" "$OUT/constant" "$SHARDS"
for f in "$OUT"/constant/shard*.sql; do $FUZZ/watchdog.sh "$BUILD" "$f" "$STALL" & done
wait
python3 $FUZZ/fuzz.py flat "$OUT/functions.json" "$OUT/constant" "$OUT/flat"
for f in "$OUT"/flat/flat*.sql; do $FUZZ/watchdog.sh "$BUILD" "$f" "$STALL" & done
wait
python3 $FUZZ/fuzz.py check "$OUT/functions.json" "$OUT/constant" "$OUT/flat" | tee "$OUT/report.txt"
