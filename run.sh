#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}${CYAN}╔══════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║        Curly Train Runner        ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════╝${NC}"
echo ""

# ── python venv ───────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo -e "${CYAN}Creating Python virtual environment...${NC}"
  python3 -m venv .venv
fi

echo -e "${CYAN}Activating venv...${NC}"
source .venv/bin/activate

echo -e "${CYAN}Installing Python deps...${NC}"
pip install -q -r requirements.txt

# ── frontend deps ─────────────────────────────────────────────────────────────
echo -e "${CYAN}Installing frontend deps...${NC}"
cd "$ROOT/frontend"
npm install --silent
cd "$ROOT"

echo ""
echo -e "${GREEN}Starting backend  → http://localhost:8000${NC}"
echo -e "${GREEN}Starting frontend → http://localhost:5173${NC}"
echo ""
echo -e "${BOLD}Press Ctrl+C to stop both servers.${NC}"
echo ""

# ── trap to kill both on exit ─────────────────────────────────────────────────
cleanup() {
  echo -e "\n${RED}Shutting down...${NC}"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  echo -e "${RED}Bye.${NC}"
}
trap cleanup EXIT INT TERM

# ── launch backend ────────────────────────────────────────────────────────────
uvicorn backend.main:app --host 0.0.0.0 --port 8000 2>&1 | sed "s/^/[${BOLD}backend${NC}] /" &
BACKEND_PID=$!

# ── launch frontend ───────────────────────────────────────────────────────────
cd "$ROOT/frontend"
npm run dev 2>&1 | sed "s/^/[${BOLD}frontend${NC}] /" &
FRONTEND_PID=$!
cd "$ROOT"

# ── wait ─────────────────────────────────────────────────────────────────────
wait "$BACKEND_PID" "$FRONTEND_PID"
