#!/bin/bash
# Build script for PINN from Scratch
# Usage:
#   ./build.sh              — build the entire course
#   ./build.sh 3            — build unit 3 only
#   ./build.sh -o 3         — build unit 3 and open in browser
#   ./build.sh serve        — serve locally at http://localhost:8080
#   ./build.sh clean        — remove all rendered output
#   ./build.sh execute      — run all sidecar scripts (units/*/scripts/*.{py,jl})
#                             and save outputs to units/*/output/. Use this for
#                             slow / Python code that won't run inside Quarto.
#   ./build.sh execute 3    — run sidecar scripts for unit 3 only

set -e
export GKSwstype=100

# Kill any existing Quarto Julia daemon so it picks up
# the GKSwstype env var on next start
pkill -f "julia.*quarto" 2>/dev/null || true

ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE="$ROOT/_site"
VENV="$ROOT/.venv"

# --- execute: run sidecar scripts and save wrapped outputs ---
run_script() {
    local script="$1"
    local lang="$2"   # "python" or "julia"
    local unit_dir
    unit_dir="$(dirname "$(dirname "$script")")"
    local out_dir="$unit_dir/output"
    mkdir -p "$out_dir"
    local stem
    stem="$(basename "$script")"
    stem="${stem%.*}"
    local out_md="$out_dir/${stem}.md"

    echo ">>> [$lang] $script"
    local raw
    raw="$out_dir/${stem}.raw"
    if [ "$lang" = "python" ]; then
        ( cd "$unit_dir" && "$VENV/bin/python" "scripts/$(basename "$script")" ) > "$raw" 2>&1 || true
    else
        ( cd "$unit_dir" && julia --project="$ROOT" "scripts/$(basename "$script")" ) > "$raw" 2>&1 || true
    fi

    {
        echo '::: {.cell-output .cell-output-stdout}'
        echo '```text'
        cat "$raw"
        echo '```'
        echo ':::'
    } > "$out_md"
    rm -f "$raw"
}

if [ "$1" = "execute" ]; then
    shift
    UNIT_FILTER=""
    if [ -n "$1" ]; then
        UNIT_FILTER="$(printf "%02d" "$1")"
    fi

    if [ -n "$UNIT_FILTER" ]; then
        SCRIPT_GLOB="$ROOT/units/unit_${UNIT_FILTER}/scripts"
    else
        SCRIPT_GLOB="$ROOT/units/unit_*/scripts"
    fi

    found=0
    for d in $SCRIPT_GLOB; do
        [ -d "$d" ] || continue
        for s in "$d"/*.py; do
            [ -f "$s" ] || continue
            run_script "$s" python
            found=1
        done
        for s in "$d"/*.jl; do
            [ -f "$s" ] || continue
            run_script "$s" julia
            found=1
        done
    done
    if [ "$found" = "0" ]; then
        echo "No sidecar scripts found."
    else
        echo "Done. Outputs in units/*/output/."
    fi
    exit 0
fi

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
    # Build everything (full project render for TOC/navigation)
    echo "Building entire course..."
    quarto render
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
