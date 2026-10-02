#!/usr/bin/env bash
# Usage: gpu-guard.sh <container>. Pauses at >=78C, resumes <=70C, kills at >=88C, on GPU loss, or on a new Xid.
# On a kill the restart policy is set to no, so Docker does not restart it into the same fault; a supervisor decides.
# Always unpauses the container on exit, so stopping the guard never leaves it frozen.
C="$1"; PAUSED=0; PSTART=0; N=0
xid() { sudo -n dmesg 2>/dev/null | grep -c "NVRM: Xid" || true; }
emergency() { docker update --restart=no "$C" >/dev/null 2>&1; docker kill "$C" >/dev/null 2>&1; }
log() { echo "$(date +%T) guard: $*"; }
cleanup() { [ "$PAUSED" = 1 ] && docker unpause "$C" >/dev/null 2>&1; }
trap cleanup EXIT
trap 'exit 143' INT TERM
BASE=$(xid)
while docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q true; do
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  case "$T" in ''|*[!0-9]*) log "no valid GPU temperature, killing $C"; emergency; exit 2;; esac
  if [ $(( N % 15 )) = 0 ] && [ "$(xid)" -gt "$BASE" ]; then log "new Xid in dmesg, killing $C"; emergency; exit 3; fi
  N=$(( N + 1 ))
  if [ "$T" -ge 88 ]; then log "${T}C hard limit, killing $C"; emergency; exit 4; fi
  if [ "$PAUSED" = 0 ] && [ "$T" -ge 78 ]; then log "${T}C pausing"; docker pause "$C"; PAUSED=1; PSTART=$(date +%s); fi
  if [ "$PAUSED" = 1 ]; then
    if [ "$T" -le 70 ]; then log "${T}C resuming"; docker unpause "$C"; PAUSED=0
    elif [ $(( $(date +%s) - PSTART )) -gt 600 ]; then log "paused >10min, killing $C"; emergency; exit 5; fi
  fi
  sleep 2
done
log "container finished"
