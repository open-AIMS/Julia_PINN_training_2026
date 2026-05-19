#!/usr/bin/env python3
"""
Generate a real-coastline site map for the Unit 1 worked example using
OSM-derived tiles (CartoDB Voyager). Two-panel figure:
  (a) Moreton Bay regional view — Brisbane River mouth + 4 gauges
      + Moreton, North Stradbroke islands; SWE-domain rectangle overlaid.
  (b) River-mouth inset — zoomed in on Brisbane River entrance / Pile
      Light area so the source location (where ψ(t) lives) is legible.

Tile source: CartoDB Voyager (CC-BY OpenStreetMap contributors / CARTO).
Output: ../figures/site_map.png

Run via:
  ../../.venv/bin/python units/unit_01/scripts/generate_site_map.py
"""

import math, time, io, csv, json
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
FIG_DIR  = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TILE_URL  = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
HEADERS   = {"User-Agent": "PINN-workshop-unit01/0.1 (course material map)"}
TILE_SIZE = 256

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG  = "/System/Library/Fonts/Supplemental/Arial.ttf"


# --- IO --------------------------------------------------------------------

def load_gauges():
    with open(DATA_DIR / "bay_gauges.csv") as f:
        return list(csv.DictReader(f))


def load_river():
    with open(DATA_DIR / "river_source.csv") as f:
        return next(csv.DictReader(f))


def load_meta():
    with open(DATA_DIR / "bay_meta.json") as f:
        return json.load(f)


# --- Web Mercator helpers --------------------------------------------------

def deg2tile(lat, lon, z):
    lat_r = math.radians(lat)
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile2deg(xt, yt, z):
    n = 2 ** z
    lon = xt / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * yt / n)))
    return math.degrees(lat_r), lon


def fetch_tile(z, x, y):
    url = TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    time.sleep(0.08)
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def build_map(north, south, east, west, z):
    x_nw, y_nw = deg2tile(north, west, z)
    x_se, y_se = deg2tile(south, east, z)
    x_min, x_max = math.floor(x_nw), math.ceil(x_se)
    y_min, y_max = math.floor(y_nw), math.ceil(y_se)
    nx, ny = x_max - x_min, y_max - y_min
    canvas = Image.new("RGB", (nx * TILE_SIZE, ny * TILE_SIZE), "white")
    for ix in range(x_min, x_max):
        for iy in range(y_min, y_max):
            try:
                tile = fetch_tile(z, ix, iy)
                canvas.paste(tile, ((ix - x_min) * TILE_SIZE,
                                    (iy - y_min) * TILE_SIZE))
            except Exception as e:
                print(f"WARN: tile {z}/{ix}/{iy}: {e}")
    lat_n, lon_w = tile2deg(x_min, y_min, z)
    lat_s, lon_e = tile2deg(x_max, y_max, z)
    return canvas, (lon_w, lat_n, lon_e, lat_s)


def _y_merc(lat_d):
    lat_r = math.radians(lat_d)
    return math.log(math.tan(math.pi / 4 + lat_r / 2))


def lonlat_to_px(lon, lat, ext, im_size):
    lon_w, lat_n, lon_e, lat_s = ext
    W, H = im_size
    y_top, y_bot = _y_merc(lat_n), _y_merc(lat_s)
    px = (lon - lon_w) / (lon_e - lon_w) * W
    py = (y_top - _y_merc(lat)) / (y_top - y_bot) * H
    return px, py


def crop_to_bbox(im, ext, north, south, east, west):
    x_l, y_t = lonlat_to_px(west, north, ext, im.size)
    x_r, y_b = lonlat_to_px(east, south, ext, im.size)
    im2 = im.crop((max(0, int(x_l)), max(0, int(y_t)),
                   min(im.size[0], int(x_r)), min(im.size[1], int(y_b))))
    return im2, (west, north, east, south)


# --- Overlays --------------------------------------------------------------

def overlay_domain_box(im, ext, meta, color=(40, 70, 200), width=3):
    """Draw the SWE-domain rectangle (lat/lon-aligned)."""
    draw = ImageDraw.Draw(im, "RGBA")
    lon_w, lat_s = meta["lon_min"], meta["lat_min"]
    lon_e = meta["lon_max"]
    lat_n = meta["lat_max"]
    corners = [(lon_w, lat_n), (lon_e, lat_n),
               (lon_e, lat_s), (lon_w, lat_s), (lon_w, lat_n)]
    pts = [lonlat_to_px(lo, la, ext, im.size) for (lo, la) in corners]
    for k in range(len(pts) - 1):
        draw.line([pts[k], pts[k + 1]], fill=color + (220,), width=width)
    # label the box (top-left, inside)
    tlx, tly = pts[0]
    try:
        f = ImageFont.truetype(FONT_BOLD, 13)
    except OSError:
        f = ImageFont.load_default()
    label = "SWE domain (50 × 95 km, 1 km grid)"
    bb = draw.textbbox((tlx + 6, tly + 6), label, font=f)
    draw.rectangle((bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2),
                   fill=(255, 255, 255, 220))
    draw.text((tlx + 6, tly + 6), label, fill=color, font=f)
    return im


def overlay_gauges_and_source(im, ext, gauges, river, label_side="auto"):
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        f_title = ImageFont.truetype(FONT_BOLD, 15)
        f_sub   = ImageFont.truetype(FONT_REG, 12)
    except OSError:
        f_title = f_sub = ImageFont.load_default()
    W, _ = im.size

    # gauges
    for s in gauges:
        lat = float(s["lat"])
        lon = float(s["lon"])
        px, py = lonlat_to_px(lon, lat, ext, im.size)
        r = 8
        draw.ellipse((px - r, py - r, px + r, py + r),
                     fill=(240, 200, 40), outline=(0, 0, 0), width=2)
        title = f'{s["gauge_id"]} — {s["name"]}'
        blurb = s["blurb"]
        flip_left = label_side == "left" or (label_side == "auto" and px > 0.55 * W)
        for j, (text, font) in enumerate([(title, f_title), (blurb, f_sub)]):
            if flip_left:
                tb = draw.textbbox((0, 0), text, font=font)
                tw = tb[2] - tb[0]
                tx = px - 12 - tw
            else:
                tx = px + 12
            ty = py - 10 + j * 17
            bb = draw.textbbox((tx, ty), text, font=font)
            draw.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2),
                           fill=(255, 255, 255, 225))
            draw.text((tx, ty), text, fill=(20, 20, 20), font=font)

    # river source — red triangle
    lat = float(river["lat"])
    lon = float(river["lon"])
    px, py = lonlat_to_px(lon, lat, ext, im.size)
    r = 11
    draw.polygon([(px, py - r), (px - r, py + r * 0.85),
                  (px + r, py + r * 0.85)],
                 fill=(220, 50, 50), outline=(0, 0, 0))
    title = f'{river["name"]}'
    blurb = "ψ(t) — unknown surge inflow"
    flip_left = px > 0.55 * W
    for j, (text, font) in enumerate([(title, f_title), (blurb, f_sub)]):
        if flip_left:
            tb = draw.textbbox((0, 0), text, font=font)
            tw = tb[2] - tb[0]
            tx = px - 14 - tw
        else:
            tx = px + 14
        ty = py - 10 + j * 17
        bb = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2),
                       fill=(255, 230, 230, 235))
        draw.text((tx, ty), text, fill=(140, 20, 20), font=font)

    return im


def scale_to_h(im, h):
    w = max(1, int(im.size[0] / im.size[1] * h))
    return im.resize((w, h), Image.LANCZOS)


# --- Main ------------------------------------------------------------------

def main():
    gauges = load_gauges()
    river  = load_river()
    meta   = load_meta()

    # Regional view — tight crop on the SWE domain (50 × 95 km) with a
    # small buffer to the east so Moreton Is.'s ocean-facing side is visible.
    REG = dict(north=-26.97, south=-27.90, west=153.00, east=153.60)
    # Inset — Brisbane River entrance / Luggage Point, zoomed in on
    # where ψ(t) is imposed (G1 is at Mud Island, well outside this box).
    INSET = dict(north=-27.28, south=-27.50, west=153.05, east=153.24)

    print("Fetching regional tiles (zoom 10)…")
    reg_im, reg_ext = build_map(z=10, **REG)
    reg_im, reg_ext = crop_to_bbox(reg_im, reg_ext, **REG)
    overlay_domain_box(reg_im, reg_ext, meta)
    overlay_gauges_and_source(reg_im, reg_ext, gauges, river)

    print("Fetching river-mouth tiles (zoom 12)…")
    ins_im, ins_ext = build_map(z=12, **INSET)
    ins_im, ins_ext = crop_to_bbox(ins_im, ins_ext, **INSET)
    # No gauges plotted in the inset — G1 (Mud Island) is now ~18 km
    # north of the inset bounding box.  The inset shows the river
    # mouth only.
    overlay_gauges_and_source(ins_im, ins_ext, [], river, label_side="left")

    target_h = 760
    reg_im   = scale_to_h(reg_im, target_h)
    ins_im   = scale_to_h(ins_im, target_h)
    gap      = 24
    title_h  = 38
    foot_h   = 32
    total_w  = reg_im.size[0] + ins_im.size[0] + gap
    total_h  = target_h + title_h + foot_h
    composite = Image.new("RGB", (total_w, total_h), "white")
    composite.paste(reg_im, (0,                      title_h))
    composite.paste(ins_im, (reg_im.size[0] + gap,   title_h))

    draw = ImageDraw.Draw(composite)
    try:
        f_top = ImageFont.truetype(FONT_BOLD, 22)
        f_cap = ImageFont.truetype(FONT_REG,  13)
        f_att = ImageFont.truetype(FONT_REG,  11)
    except OSError:
        f_top = f_cap = f_att = ImageFont.load_default()

    draw.text((12, 8),
        "Unit 1 worked example — Brisbane River surge in Moreton Bay",
        fill=(20, 20, 20), font=f_top)

    cap1 = "(a) regional view · SWE domain (blue) · 4 tide gauges · river-mouth source"
    cap2 = "(b) Brisbane River mouth inset · Luggage Point · source cell"
    b1 = draw.textbbox((0, 0), cap1, font=f_cap)
    b2 = draw.textbbox((0, 0), cap2, font=f_cap)
    draw.text((reg_im.size[0] // 2 - (b1[2] - b1[0]) // 2,
               title_h + target_h + 5), cap1, fill=(40, 40, 40), font=f_cap)
    draw.text((reg_im.size[0] + gap + ins_im.size[0] // 2 - (b2[2] - b2[0]) // 2,
               title_h + target_h + 5), cap2, fill=(40, 40, 40), font=f_cap)

    attr = "© OpenStreetMap contributors, © CARTO (Voyager tiles)"
    b3 = draw.textbbox((0, 0), attr, font=f_att)
    draw.text((total_w - (b3[2] - b3[0]) - 10, total_h - 16),
              attr, fill=(95, 95, 95), font=f_att)

    outpath = FIG_DIR / "site_map.png"
    composite.save(outpath)
    print(f"wrote {outpath}  ({composite.size[0]}×{composite.size[1]} px)")


if __name__ == "__main__":
    main()
