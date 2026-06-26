#!/usr/bin/env bash
# Build the course notes as a single PDF booklet — WITHOUT re-running any Julia.
#
#   ./booklet/build.sh
#
# How it avoids recompute: Quarto's freeze is format-specific (html.json vs
# tex.json), but the freeze *hash* is computed from the source, not the format.
# So we copy each committed html.json execution result into the tex.json slot
# (stripping HTML-only includes/preserve); freeze:auto then sees a matching
# hash and reuses it, and the PDF render executes zero Julia cells.
#
# Requires: quarto, xelatex (TeX Live), rsvg-convert (brew install librsvg),
#           JuliaMono (brew install --cask font-juliamono).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Locate JuliaMono (carries the Unicode glyphs used in Julia code).
FONTDIR=""
for d in "$HOME/Library/Fonts" "/Library/Fonts"; do
  [ -f "$d/JuliaMono-Regular.ttf" ] && FONTDIR="$d" && break
done
[ -z "$FONTDIR" ] && { echo "JuliaMono not found — run: brew install --cask font-juliamono"; exit 1; }
echo "Using JuliaMono from: $FONTDIR"
cat > booklet/header.tex <<TEX
\setmonofont{JuliaMono-}[
  Path=$FONTDIR/, Extension=.ttf,
  UprightFont=*Regular, BoldFont=*Bold,
  ItalicFont=*RegularItalic, BoldItalicFont=*BoldItalic,
  Scale=0.82]
TEX

# 2. Reuse the HTML freeze for the PDF (no Julia recompute).
python3 - <<'PY'
import json, glob
n = 0
for hp in glob.glob('_freeze/**/execute-results/html.json', recursive=True):
    tp = hp[:-len('html.json')] + 'tex.json'
    d = json.load(open(hp)); r = d.get('result', {})
    r['includes'] = {}                       # drop HTML head/deps
    if 'preserve' in r: r['preserve'] = {}    # drop preserved raw-HTML blocks
    if 'engineDependencies' in r: r['engineDependencies'] = {}
    json.dump(d, open(tp, 'w')); n += 1
print(f"synced {n} freeze results html.json -> tex.json (no recompute)")
PY

# 3. Render the book.
quarto render --profile booklet --to pdf
echo "Booklet written to _booklet/"
