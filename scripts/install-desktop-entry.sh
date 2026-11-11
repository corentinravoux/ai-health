#!/usr/bin/env bash
# Add ai-health to the desktop app menu (~/.local/share/applications).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${AI_HEALTH_EXE:-$HOME/Software/venv/bin/ai-health}"
[[ -x "$EXE" ]] || { echo "ai-health not found at $EXE (install it into ~/Software/venv first)" >&2; exit 1; }
DEST="$HOME/.local/share/applications/ai-health.desktop"
mkdir -p "$(dirname "$DEST")"
cat > "$DEST" <<DESKTOP
[Desktop Entry]
Type=Application
Name=ai-health
GenericName=Local LLM monitor
Comment=Watch a local llama.cpp LLM (tokens/s, VRAM, RAM) and explore GGUF models
Exec=$EXE
Icon=$REPO/ai_health/static/icon.svg
Terminal=false
Categories=System;Monitor;Development;
StartupWMClass=ai-health
DESKTOP
update-desktop-database "$(dirname "$DEST")" 2>/dev/null || true
echo "installed $DEST"
