"""Figure 7: global distribution of CrustNet monitoring sites and representative field activities.

a  CrustNet sites (n = 64) over the PACE OCI four-index composite of Fig. 4a (May 2025 monthly composite,
   drylands only, same colour key); all sites are shown, including those outside the drylands
b  representative field activities at CrustNet sites (three photographs)

Data:
  1_data/crustnet_sites.csv          site codes and coordinates (columns: code, lat, lon), extracted from the
                                     CrustNet node map (https://crustnet.org/CrustNet_Map5.html -> embedded
                                     leaflet/htmlwidgets payload)
  PACE composite and dryland mask    read through plot_fig4.py (its display cache), so colours match Fig. 4a
  1_data/7_Fig7_data/image1-3.png    field photographs

Sized for Nature: 183 mm wide (double column), Arial, 1000 dpi PNG plus vector PDF.

Run with the Fujiang env:
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_fig7.py
"""
import csv
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import plot_fig3 as f3     # spectral-library sites (coordinates from Karnieli et al., Table 1)
import plot_fig4 as f4

DATA_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/0_src_code/1_data"
PHOTO_DIR = os.path.join(DATA_DIR, "7_Fig7_data")
FIG_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/figures"
SITES_CSV = os.path.join(DATA_DIR, "crustnet_sites.csv")

# field photographs: (file, square crop box as fractions of the image (x0, y0, x1, y1))
PHOTOS = [
    ("image1.png", (0.0, 0.20, 1.0, 0.95)),
    ("image2.png", (0.0, 0.04, 1.0, 0.79)),
    ("image3.png", (0.0, 0.05, 1.0, 0.80)),
]

SITE_COLOR = "#d7301f"
SITE_EFFECTS = [pe.Stroke(linewidth=1.5, foreground="#1a1a1a"), pe.Normal()]   # dark ring outside the white edge
LIB_COLOR = "#ff00ff"      # spectral-library sites of Fig. 3: magenta stars
LIB_SITES = {name: s["lonlat"] for name, s in f3.SITES.items()}
KEY_AT = (-125, -21)       # lon, lat of the colour key's centre (empty South Pacific); two-line site legend below it

MM = 1 / 25.4
FIG_W = 183 * MM
NATURE_RC = dict(f4.NATURE_RC, **{"font.size": 6.5, "legend.fontsize": 6.5})
PANEL_LETTER_SIZE = 8
PANEL_TITLE_SIZE = 8
MAP_LABEL_SIZE = 6.5

# layout (inches)
SIDE = 0.30            # latitude labels beside the globe
HEADING = 0.26         # panel letter + title above a row
LON_LABELS = 0.17      # longitude labels under the globe
GAP = 0.06             # gap between photographs


def heading(fig, x, y, letter, title):
    w, h = fig.get_size_inches()
    fig.text(x / w, y / h, letter, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="baseline")
    fig.text((x + 0.16) / w, y / h, title, fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="baseline")


def read_sites(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [float(r["lon"]) for r in rows], [float(r["lat"]) for r in rows]


def read_photo(fname, box):
    im = Image.open(os.path.join(PHOTO_DIR, fname)).convert("RGB")
    w, h = im.size
    return np.asarray(im.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h))))


def pace_composite():
    """PACE layer of Fig. 4a: drylands only, coloured with Fig. 4's key window (pooled over all platforms)."""
    data, transforms, aridity = f4.load_display_arrays()
    dry = np.isin(aridity, f4.DRYLAND_CLASSES)
    data["PACE"] = {n: np.where(dry, a, np.nan) for n, a in data["PACE"].items()}
    positions = {p: f4.square_position(d) for p, d in data.items()}
    window = f4.key_window(positions.values())
    return f4.composite_rgba(positions["PACE"], window), transforms["PACE"], aridity


def lonlat_to_inches(ax, lon, lat):
    ax.apply_aspect()
    xy = ax.projection.transform_point(lon, lat, f4.PC)
    return ax.transData.transform(xy) / ax.figure.dpi


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(NATURE_RC)
    lons, lats = read_sites(SITES_CSV)
    photos = [read_photo(f, box) for f, box in PHOTOS]
    rgba, transform, aridity = pace_composite()

    a_w = FIG_W - 2 * SIDE
    a_h = a_w / f4.global_aspect()
    tile = (a_w - 2 * GAP) / 3
    H = HEADING + a_h + LON_LABELS + HEADING + tile + 0.04
    fig = plt.figure(figsize=(FIG_W, H))
    W = FIG_W

    # a: PACE composite over drylands, CrustNet sites on top
    a_bottom = H - HEADING - a_h
    heading(fig, SIDE, H - HEADING + 0.08, "a", "Global distribution of CrustNet monitoring sites")
    ax = f4.axes_in(fig, SIDE, a_bottom, a_w, a_h, projection=f4.ROBINSON)
    f4.draw_global(ax, rgba, transform, aridity)
    f4.global_gridlines(fig, ax, size=MAP_LABEL_SIZE)
    ax.spines["geo"].set_linewidth(1.0)
    ax.scatter(lons, lats, transform=f4.PC, s=16, c=SITE_COLOR, edgecolor="white", linewidth=0.5, zorder=8,
               path_effects=SITE_EFFECTS)
    lib_lons, lib_lats = zip(*LIB_SITES.values())
    ax.scatter(lib_lons, lib_lats, transform=f4.PC, s=48, marker="*", c=LIB_COLOR, edgecolor="black",
               linewidth=0.5, zorder=9)
    print("  library sites:", ", ".join(LIB_SITES))

    # colour key (same as Fig. 4) in the empty South Pacific, site legend centred below it
    key = 0.55
    kx, ky = lonlat_to_inches(ax, *KEY_AT)
    f4.draw_key(f4.axes_in(fig, kx - key / 2, ky - key / 2, key, key), fontsize=6)
    site = Line2D([], [], ls="none", marker="o", markersize=4.5, markerfacecolor=SITE_COLOR,
                  markeredgecolor="white", markeredgewidth=0.5, path_effects=SITE_EFFECTS)
    lib = Line2D([], [], ls="none", marker="*", markersize=7.5, markerfacecolor=LIB_COLOR, markeredgecolor="black",
                 markeredgewidth=0.5)
    fig.legend(handles=[site, lib],
               labels=[f"CrustNet site (n = {len(lons)})", f"Spectral library site (n = {len(LIB_SITES)})"],
               loc="upper center", bbox_to_anchor=((kx + 0.15) / W, (ky - key / 2 - 0.16) / H), frameon=False,
               handlelength=1.0, handletextpad=0.4, labelspacing=0.5, borderaxespad=0, borderpad=0)

    # b: field photographs, aligned with the map
    b_top = a_bottom - LON_LABELS - HEADING
    heading(fig, SIDE, b_top + 0.08, "b", "Representative field observations and biocrust sampling")
    for k, img in enumerate(photos):
        ph = f4.axes_in(fig, SIDE + k * (tile + GAP), b_top - tile, tile, tile)
        ph.imshow(img)
        ph.set_xticks([])
        ph.set_yticks([])
        for spine in ph.spines.values():
            spine.set_linewidth(0.6)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"Fig7.{ext}")
        fig.savefig(out, dpi=1000, facecolor="white")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
