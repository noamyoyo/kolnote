#!/usr/bin/env bash
# Usage: gpu-guard.sh <container>. Pauses at >=78C, resumes <=70C, kills at >=88C, on GPU loss, or on a new Xid.
C="$1"; PAUSED=0; PSTART=0
xid() { sudo -n dmesg 2>/dev/null | grep -c "NVRM: Xid" || true; }
BASE=$(xid)
log() { echo "$(date +%T) guard: $*"; }
while docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q true; do
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null) || { log "nvidia-smi failed, killing $C"; docker kill "$C"; exit 2; }
  if [ "$(xid)" != "$BASE" ]; then log "new Xid in dmesg, killing $C"; docker kill "$C"; exit 3; fi
  if [ "$T" -ge 88 ]; then log "${T}C hard limit, killing $C"; docker kill "$C"; exit 4; fi
  if [ "$PAUSED" = 0 ] && [ "$T" -ge 78 ]; then log "${T}C pausing"; docker pause "$C"; PAUSED=1; PSTART=$(date +%s); fi
  if [ "$PAUSED" = 1 ]; then
    if [ "$T" -le 70 ]; then log "${T}C resuming"; docker unpause "$C"; PAUSED=0
    elif [ $(( $(date +%s) - PSTART )) -gt 600 ]; then log "paused >10min, killing $C"; docker kill "$C"; exit 5; fi
  fi
  sleep 2
done
log "container finished"
