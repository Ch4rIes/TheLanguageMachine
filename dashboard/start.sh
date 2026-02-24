#!/usr/bin/env bash
# Start the CS336 experiment dashboard (backend + frontend)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "=== CS336 Experiment Dashboard ==="
echo ""

# Install backend dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "[backend] Installing dependencies..."
    pip install -r "$BACKEND_DIR/requirements.txt"
fi

# Start backend
echo "[backend] Starting FastAPI on http://localhost:8000 ..."
cd "$BACKEND_DIR"
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "[frontend] Starting Vite on http://localhost:5173 ..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Dashboard running:"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait
