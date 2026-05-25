#!/usr/bin/env bash
# deploy.sh — Sync project to Pi, restart TARS, tail the log.
#
# Usage:
#   ./deploy.sh              # deploy and tail log
#   ./deploy.sh --no-tail    # deploy and restart, don't tail
#   ./deploy.sh --stop       # stop TARS without redeploying
#
# Prerequisites:
#   - SSH key auth configured: ssh-copy-id pi@tars.local
#   - tmux installed on Pi: sudo apt-get install tmux
#   - TARS_HOST set below

set -euo pipefail

# ── Configure this ──────────────────────────────────────────────────────────
TARS_HOST="${TARS_HOST:-pi@192.168.0.20}"  # override with: TARS_HOST=pi@192.168.1.42 ./deploy.sh
REMOTE_DIR="~/tars-mini"
TMUX_SESSION="tars"
LOG_FILE="$REMOTE_DIR/tars.log"
VENV_PYTHON="$REMOTE_DIR/.venv/bin/python"
# ────────────────────────────────────────────────────────────────────────────

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()    { echo -e "${YELLOW}[deploy]${NC} $*"; }
die()     { echo -e "${RED}[deploy] ERROR:${NC} $*" >&2; exit 1; }

# ── --stop shortcut ──────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    info "Stopping TARS on $TARS_HOST…"
    ssh "$TARS_HOST" "pkill -f tars_pi.py || true; tmux kill-session -t $TMUX_SESSION 2>/dev/null || true"
    info "Stopped."
    exit 0
fi

# ── Verify SSH reachable ─────────────────────────────────────────────────────
info "Checking SSH connection to $TARS_HOST…"
ssh -o ConnectTimeout=5 "$TARS_HOST" "echo ok" > /dev/null \
    || die "Cannot reach $TARS_HOST. Check hostname/IP and SSH key auth."

# ── Create remote directory ──────────────────────────────────────────────────
info "Ensuring $REMOTE_DIR exists on Pi…"
ssh "$TARS_HOST" "mkdir -p $REMOTE_DIR"

# ── Sync files ───────────────────────────────────────────────────────────────
# Note: rsync is unavailable on this Windows machine (Git Bash); using scp instead.
info "Syncing files to $TARS_HOST:$REMOTE_DIR…"
for f in tars_pi.py requirements_pi.txt deploy.sh CLAUDE.md; do
    [[ -f "$f" ]] && scp -q "$f" "$TARS_HOST:$REMOTE_DIR/$f"
done
info "Sync complete."

# ── Install/update Python deps if requirements.txt changed ───────────────────
info "Checking Python venv…"
ssh "$TARS_HOST" bash << 'REMOTE'
set -e
cd ~/tars-mini

# Ensure portaudio is installed (needed for pyaudio to build)
if ! dpkg -l portaudio19-dev &>/dev/null; then
    echo "[pi] Installing portaudio19-dev…"
    sudo apt-get install -y portaudio19-dev -q
fi

# Use system python3 (3.5, has SSL) not the custom-built 3.11 (no SSL)
PYTHON=/usr/bin/python3

if [ ! -f .venv/bin/python ]; then
    echo "[pi] Creating virtual environment…"
    # --system-site-packages: inherit python3-pyaudio from apt (pip can't use SSL on this OS)
    $PYTHON -m venv --system-site-packages .venv
fi

echo "[pi] Verifying pyaudio is available…"
.venv/bin/python -c "import pyaudio; print('[pi] pyaudio', pyaudio.__version__, 'OK')"
echo "[pi] Dependencies OK."
REMOTE

# ── Stop existing TARS process ───────────────────────────────────────────────
info "Stopping any running TARS process…"
ssh "$TARS_HOST" "pkill -f tars_pi.py 2>/dev/null || true; sleep 1"

# ── Start TARS Pi client in a tmux session ───────────────────────────────────
info "Starting TARS Pi client in tmux session '$TMUX_SESSION'…"
ssh "$TARS_HOST" bash << REMOTE
set -e
cd ~/tars-mini

tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

tmux new-session -d -s $TMUX_SESSION -x 220 -y 50 \
    "bash -lc 'cd ~/tars-mini && .venv/bin/python tars_pi.py 2>&1 | tee tars.log'; bash"

echo "[pi] TARS Pi client started in tmux session '$TMUX_SESSION'."
REMOTE

info "TARS deployed and running."
info ""
info "Useful commands:"
info "  Watch log:     ssh $TARS_HOST 'tail -f $LOG_FILE'"
info "  Attach tmux:   ssh -t $TARS_HOST 'tmux attach -t $TMUX_SESSION'"
info "  Stop TARS:     ./deploy.sh --stop"
info ""

# ── Tail log (unless --no-tail) ───────────────────────────────────────────────
if [[ "${1:-}" != "--no-tail" ]]; then
    info "Tailing log (Ctrl+C to stop watching — TARS keeps running)…"
    echo ""
    sleep 1   # give TARS a moment to start writing
    ssh "$TARS_HOST" "tail -f $LOG_FILE" || true
fi
