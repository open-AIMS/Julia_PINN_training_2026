#!/bin/bash
# Build script for PINN from Scratch
# Usage:
#   ./build.sh              — build the entire course
#   ./build.sh 3            — build unit 3 only
#   ./build.sh -o 3         — build unit 3 and open in browser
#   ./build.sh serve        — serve locally at http://localhost:8080
#   ./build.sh clean        — remove all rendered output

set -e
export GKSwstype=100

# Kill any existing Quarto Julia daemon so it picks up
# the GKSwstype env var on next start
pkill -f "julia.*quarto" 2>/dev/null || true

ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE="$ROOT/_site"

# Parse -o flag
OPEN=false
if [ "$1" = "-o" ]; then
    OPEN=true
    shift
fi

open_files() {
    if $OPEN; then
        for f in "$@"; do
            open "$f"
        done
    fi
}

if [ "$1" = "serve" ]; then
    if [ ! -d "$SITE" ]; then
        echo "Error: $SITE does not exist. Run ./build.sh first." >&2
        exit 1
    fi
    echo "Serving at http://localhost:8080 (Ctrl-C to stop)"
    open "http://localhost:8080"
    python3 -m http.server -d "$SITE" 8080
    exit 0
fi

if [ "$1" = "clean" ]; then
    echo "Cleaning $SITE ..."
    rm -rf "$SITE"
    echo "Done."
    exit 0
fi

if [ $# -eq 0 ]; then
    # Build everything
    echo "Building entire course..."
    quarto render "$ROOT/index.qmd"
    for f in "$ROOT"/units/unit_*/unit_*.qmd "$ROOT"/units/appendix_*/appendix_*.qmd; do
        [ -f "$f" ] || continue
        echo "  Rendering $(basename "$f") ..."
        quarto render "$f"
    done
    echo "Done. Output in $SITE"
    if $OPEN; then
        open "$SITE"
    fi

elif [ $# -eq 1 ]; then
    # Build a single unit
    UNIT=$(printf "%02d" "$1")
    FILE="$ROOT/units/unit_$UNIT/unit_$UNIT.qmd"
    if [ ! -f "$FILE" ]; then
        echo "Error: $FILE does not exist." >&2
        exit 1
    fi
    echo "Building unit $UNIT ..."
    quarto render "$FILE"
    OUTPUT="$SITE/units/unit_$UNIT/unit_$UNIT.html"
    echo "Done."
    open_files "$OUTPUT"

else
    echo "Usage: ./build.sh [-o] [unit]" >&2
    echo "  ./build.sh              build entire course" >&2
    echo "  ./build.sh 3            build unit 3" >&2
    echo "  ./build.sh -o 3         build unit 3 and open in browser" >&2
    echo "  ./build.sh serve        serve locally at http://localhost:8080" >&2
    echo "  ./build.sh clean        remove rendered output" >&2
    exit 1
fi
