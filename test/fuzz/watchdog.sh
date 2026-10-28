#!/bin/bash
# Usage: watchdog.sh <build_dir> <shard.sql> <stall seconds>
# Runs a shard through the DuckDB CLI. When output stalls, the running statement is recorded as hung, the CLI
# is killed and the shard resumes after that statement. A crash (signal) is recorded and the shard also resumes.
BUILD=$1; SQL=$2; STALL=$3
# Line buffering keeps each error next to the statement that raised it
LINE_BUFFERED=$(command -v stdbuf > /dev/null && echo "stdbuf -oL -eL")
LOG=${SQL%.sql}.log; : > "$LOG"
HEADER=$(head -1 "$SQL" | sed "s#{extension}#$BUILD/extension/stochastic/stochastic.duckdb_extension#")
REST=$(mktemp); tail -n +2 "$SQL" > "$REST"
while [ -s "$REST" ]; do
  PART=$(mktemp); { echo "$HEADER"; cat "$REST"; } > "$PART"
  OUT=$(mktemp)
  $LINE_BUFFERED $BUILD/duckdb -unsigned -noheader -list < "$PART" > "$OUT" 2>&1 &
  PID=$!; LAST=-1; IDLE=0; STATUS=
  while kill -0 $PID 2>/dev/null; do
    sleep 1; SIZE=$(wc -c < "$OUT")
    if [ "$SIZE" = "$LAST" ]; then IDLE=$((IDLE + 1)); else IDLE=0; LAST=$SIZE; fi
    if [ $IDLE -ge $STALL ]; then kill -9 $PID; wait $PID 2>/dev/null; STATUS=hang; break; fi
  done
  [ -z "$STATUS" ] && { wait $PID; STATUS=$?; }
  cat "$OUT" >> "$LOG"
  CURRENT=$(grep -E '^Q[0-9]+$' "$OUT" | tail -1)
  rm -f "$PART" "$OUT"
  if [ "$STATUS" = hang ] || [ "$STATUS" -ge 128 ] 2>/dev/null; then
    [ "$STATUS" = hang ] && echo "HANG $CURRENT" >> "$LOG" || echo "CRASH $STATUS $CURRENT" >> "$LOG"
    [ -z "$CURRENT" ] && break
    # Resume after the statement that hung or crashed
    awk -v q=".print $CURRENT" 'found { print } $0 == q { skip = 1; next } skip && /^\.print Q/ { found = 1; print }' "$REST" > "$REST.next"
    mv "$REST.next" "$REST"
  else
    break
  fi
done
rm -f "$REST"
echo EXIT_STATUS=0 >> "$LOG"
