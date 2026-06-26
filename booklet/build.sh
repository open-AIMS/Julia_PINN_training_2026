#!/usr/bin/env bash
# Build the course notes as a single PDF booklet.
#   ./booklet/build.sh            # render-from-cache where possible; re-executes Julia units lacking a tex.json freeze
# Requires: quarto, xelatex (TeX Live), rsvg-convert (brew install librsvg),
#           JuliaMono font (brew install --cask font-juliamono).
set -euo pipefail
cd "$(dirname "$0")/.."

# Locate JuliaMono (macOS user/system font dirs).
FONTDIR=""
for d in "$HOME/Library/Fonts" "/Library/Fonts"; do
  [ -f "$d/JuliaMono-Regular.ttf" ] && FONTDIR="$d" && break
done
[ -z "$FONTDIR" ] && { echo "JuliaMono not found — run: brew install --cask font-juliamono"; exit 1; }
echo "Using JuliaMono from: $FONTDIR"

# Generate the font header (monospace font that carries Julia's Unicode glyphs).
cat > booklet/header.tex <<TEX
\setmonofont{JuliaMono-}[
  Path=$FONTDIR/, Extension=.ttf,
  UprightFont=*Regular, BoldFont=*Bold,
  ItalicFont=*RegularItalic, BoldItalicFont=*BoldItalic,
  Scale=0.82]
TEX

quarto render --profile booklet --to pdf
echo "Booklet written to _booklet/"
