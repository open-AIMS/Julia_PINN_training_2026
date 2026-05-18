#!/usr/bin/env python3
"""
Generate a real-coastline site map for the capstone using OSM-derived
tiles (CartoDB Voyager).  Two-panel figure:
  (a) regional view of all three mooring sites along the central GBR
      cross-shelf transect (Cleveland Bay → Davies Reef → Myrmidon Reef);
  (b) inshore inset zoomed on Cleveland Bay so the coast around Site A
      is legible.

Tile source: CartoDB Voyager (CC-BY OpenStreetMap contributors / CARTO).
Output: ../figures/site_map.png

Run via:
  ../../.venv/bin/python units/unit_09/scripts/generate_site_map.py
"""

import math, time, io
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont

SITES = [
    dict(id="A", name="Cleveland Bay",  lat=-19.200, lon=146.810,
         blurb="inshore · 15 m · tidal mixing"),
    dict(id="B", name="Davies Reef",    lat=-18.830, lon=147.647,
         blurb="mid-shelf · 60 m · classical thermocline"),
    dict(id="C", name="Myrmidon Reef",  lat=-18.270, lon=147.390,
         blurb="outer shelf · 100 m · advection-dominated"),
]

TILE_URL  = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
HEADERS   = {"User-Agent": "AIMS-PINN-workshop/0.1 (course material map)"}
TILE_SIZE = 256
HERE      = Path(__file__).resolve().parent

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG  = "/System/Library/Fonts/Supplemental/Arial.ttf"


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


def overlay_sites(im, ext, sites):
    """Draw each site marker, auto-flipping the label to the left of the
    marker if the marker sits in the right third of the image so the text
    doesn't run off the edge."""
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        f_title = ImageFont.truetype(FONT_BOLD, 17)
        f_sub   = ImageFont.truetype(FONT_REG, 13)
    except OSError:
        f_title = ImageFont.load_default()
        f_sub   = f_title
    W, _ = im.size
    for s in sites:
        px, py = lonlat_to_px(s["lon"], s["lat"], ext, im.size)
        r = 9
        draw.ellipse((px - r, py - r, px + r, py + r),
                     fill=(220, 30, 30), outline=(0, 0, 0), width=2)
        title = f'Site {s["id"]} — {s["name"]}'
        # decide side: if the marker is in the right ~40% of the image,
        # flip the label to the left.
        flip_left = px > 0.6 * W
        for j, (text, font) in enumerate([(title, f_title), (s["blurb"], f_sub)]):
            if flip_left:
                # measure width to right-anchor the text
                tb = draw.textbbox((0, 0), text, font=font)
                tw = tb[2] - tb[0]
                tx = px - 14 - tw
            else:
                tx = px + 14
            ty = py - 12 + j * 19
            bbox = draw.textbbox((tx, ty), text, font=font)
            draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                           fill=(255, 255, 255, 225))
            draw.text((tx, ty), text, fill=(20, 20, 20), font=font)
    return im


REG_BOUNDS   = dict(north=-18.00, south=-19.55, west=146.30, east=148.00)
INSET_BOUNDS = dict(north=-19.05, south=-19.35, west=146.62, east=147.05)

print("Fetching regional tiles (zoom 9)…")
reg_im, reg_ext = build_map(z=9, **REG_BOUNDS)
reg_im, reg_ext = crop_to_bbox(reg_im, reg_ext, **REG_BOUNDS)
overlay_sites(reg_im, reg_ext, SITES)

print("Fetching inshore tiles (zoom 11)…")
inset_im, inset_ext = build_map(z=11, **INSET_BOUNDS)
inset_im, inset_ext = crop_to_bbox(inset_im, inset_ext, **INSET_BOUNDS)
overlay_sites(inset_im, inset_ext,
              [s for s in SITES if s["id"] == "A"])


def scale_to_h(im, h):
    w = max(1, int(im.size[0] / im.size[1] * h))
    return im.resize((w, h), Image.LANCZOS)


target_h = 760
reg_im   = scale_to_h(reg_im, target_h)
inset_im = scale_to_h(inset_im, target_h)
gap      = 24
title_h  = 38
foot_h   = 32
total_w  = reg_im.size[0] + inset_im.size[0] + gap
total_h  = target_h + title_h + foot_h
composite = Image.new("RGB", (total_w, total_h), "white")
composite.paste(reg_im,   (0,                       title_h))
composite.paste(inset_im, (reg_im.size[0] + gap,    title_h))

draw = ImageDraw.Draw(composite)
try:
    f_top = ImageFont.truetype(FONT_BOLD, 22)
    f_cap = ImageFont.truetype(FONT_REG,  13)
    f_att = ImageFont.truetype(FONT_REG,  11)
except OSError:
    f_top = f_cap = f_att = ImageFont.load_default()

draw.text((12, 8),
    "Capstone mooring transect — central Great Barrier Reef",
    fill=(20, 20, 20), font=f_top)

cap1 = "(a) regional view · all three sites"
cap2 = "(b) inshore inset · Cleveland Bay (Site A)"
b1 = draw.textbbox((0, 0), cap1, font=f_cap)
b2 = draw.textbbox((0, 0), cap2, font=f_cap)
draw.text((reg_im.size[0] // 2 - (b1[2] - b1[0]) // 2,
           title_h + target_h + 5), cap1, fill=(40, 40, 40), font=f_cap)
draw.text((reg_im.size[0] + gap + inset_im.size[0] // 2 - (b2[2] - b2[0]) // 2,
           title_h + target_h + 5), cap2, fill=(40, 40, 40), font=f_cap)

attr = "© OpenStreetMap contributors, © CARTO (Voyager tiles)"
b3 = draw.textbbox((0, 0), attr, font=f_att)
draw.text((total_w - (b3[2] - b3[0]) - 10, total_h - 16),
          attr, fill=(95, 95, 95), font=f_att)

outpath = HERE / ".." / "figures" / "site_map.png"
outpath = outpath.resolve()
outpath.parent.mkdir(parents=True, exist_ok=True)
composite.save(outpath)
print(f"wrote {outpath}  ({composite.size[0]}×{composite.size[1]} px)")
