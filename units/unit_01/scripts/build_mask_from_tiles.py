#!/usr/bin/env python3
# Build the Moreton Bay land/water mask directly from rendered OpenStreetMap map
# tiles: fetch the tiles covering the SWE domain, threshold the water colour
# (#aad3df), resample to the 100x190 grid, then clean speckle and keep the main
# ocean-connected water body. Writes data/bay_mask.csv (the shape build_bay.jl
# then drapes synthetic depths + gauges onto).
#
# Needs network (tile fetch); run once, commit data/bay_mask.csv. The derived
# mask is factual geography (land vs water); tiles are © OpenStreetMap, ODbL.
#
#   python3 build_mask_from_tiles.py
import math, time, io, os, urllib.request
import numpy as np
from PIL import Image
from collections import deque

# --- domain (must match build_bay.jl) -------------------------------------
LON_MIN, LAT_MIN, LAT_REF = 153.020, -27.850, -27.40
MPD_LAT = 111_000.0
MPD_LON = 111_000.0 * math.cos(math.radians(LAT_REF))
DX = DY = 500.0
NX, NY = 100, 190
LON_MAX = LON_MIN + NX * DX / MPD_LON
LAT_MAX = LAT_MIN + NY * DY / MPD_LAT
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "bay_mask.csv")

# --- fetch the OSM tiles covering the domain ------------------------------
Z = 12
n = 2 ** Z
lon2px = lambda lon: (lon + 180) / 360 * n * 256
lat2px = lambda lat: (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n * 256
xt0, xt1 = int(lon2px(LON_MIN) // 256), int(lon2px(LON_MAX) // 256)
yt0, yt1 = int(lat2px(LAT_MAX) // 256), int(lat2px(LAT_MIN) // 256)
hdr = {"User-Agent": "PINN-course-bathymetry/1.0 (educational, one-time)"}
tiles = {}
print(f"fetching {(xt1-xt0+1)*(yt1-yt0+1)} OSM tiles at z{Z} ...")
for xt in range(xt0, xt1 + 1):
    for yt in range(yt0, yt1 + 1):
        url = f"https://tile.openstreetmap.org/{Z}/{xt}/{yt}.png"
        for _ in range(3):
            try:
                d = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=30).read()
                tiles[(xt, yt)] = np.asarray(Image.open(io.BytesIO(d)).convert("RGB"))
                break
            except Exception:
                time.sleep(1)
        time.sleep(0.05)

# --- threshold the water colour onto the grid (sub-sampled majority) ------
def is_water(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return abs(r - 170) + abs(g - 211) + abs(b - 223) < 45   # OSM water #aad3df

def colour(gpx, gpy):
    t = tiles.get((int(gpx // 256), int(gpy // 256)))
    return (0, 0, 0) if t is None else t[int(gpy) % 256, int(gpx) % 256]

mask = np.zeros((NY, NX), int)
for j in range(1, NY + 1):
    for i in range(1, NX + 1):
        wc = tot = 0
        for du in (-0.3, 0, 0.3):
            for dv in (-0.3, 0, 0.3):
                lo = LON_MIN + (i - 0.5 + du) * DX / MPD_LON
                la = LAT_MIN + (j - 0.5 + dv) * DY / MPD_LAT
                tot += 1; wc += is_water(colour(lon2px(lo), lat2px(la)))
        mask[j - 1, i - 1] = 1 if wc / tot > 0.5 else 0

# --- clean: 3x3 majority filter (twice), then keep the main water body -----
def majority(m):
    out = m.copy()
    for j in range(NY):
        for i in range(NX):
            s = c = 0
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < NY and 0 <= ii < NX:
                        s += m[jj, ii]; c += 1
            out[j, i] = 1 if s * 2 > c else 0
    return out

mask = majority(majority(mask))

# keep only water connected to the open-ocean edge (drop isolated label specks)
keep = np.zeros((NY, NX), bool); q = deque()
for j in range(NY):
    for i in (0, NX - 1):
        if mask[j, i]: keep[j, i] = True; q.append((j, i))
for i in range(NX):
    for j in (0, NY - 1):
        if mask[j, i] and not keep[j, i]: keep[j, i] = True; q.append((j, i))
while q:
    r, c = q.popleft()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < NY and 0 <= nc < NX and mask[nr, nc] and not keep[nr, nc]:
            keep[nr, nc] = True; q.append((nr, nc))
mask = keep.astype(int)

np.savetxt(OUT, mask, fmt="%d", delimiter=",")
print(f"water {100*mask.mean():.1f}%  ->  {os.path.relpath(OUT)}")
