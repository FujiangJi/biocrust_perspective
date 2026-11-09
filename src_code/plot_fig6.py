"""Figure 6: upscaling biocrust fractional cover from UAS observations to EMIT hyperspectral imagery.

a  field photographs: biocrust within a sampling frame, and close-ups of lichen and moss (representative;
   not from the UAS site)
b  UAS RGB orthomosaic and level-1 classification (Horse Thief site); a 15-m-radius detail is magnified,
   the 60-m EMIT grid is drawn on the classification, and the class fractions of one 60-m cell (a training
   target) are shown below
c  three-component colour composite of the EMIT-predicted cover (5 April 2025 scene); the arrows link the
   UAS area in b to its location in the scene. The class legend (b, d) and the ternary key (c) sit below.
d  predicted fractional cover of late-successional biocrust, light surface and vascular plants

Data:
  HT_RGB61_utm12n.tif (external drive)    UAS RGB orthomosaic, 1 cm, read directly (its overviews for the
                                          map, native pixels for the magnified detail)
  1_data/6_Fig6_data/
    HT_RF_points_level1_broad.tif         UAS level-1 classification, 2 cm (RF, points)
    HT_level1_frac_60m.tif                class fractions per 60-m EMIT cell (training targets)
    EMIT_..._level1_pca_rf_refl_indices_spatial_cover.tif   EMIT-scale predictions (5 classes)
    HT_MS_extent.geojson                  UAS footprint
    IMG_*.jpg                             field photographs
The UAS map crop and the magnified spot are those of the original Fig. 6a
(5_Upscaling_biocrust_mapping/4_UAS_classification/0_src_code/6_pixel_object_plot.py).

Sized for Nature: 183 mm wide (double column), Arial, 1000 dpi PNG plus vector PDF.

Run with the Fujiang env (the external drive must be mounted):
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_fig6.py
"""
import json
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import bounds as window_bounds, from_bounds
from pyproj import Transformer
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator

DATA_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/0_src_code/1_data/6_Fig6_data"
FIG_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/figures"

EMIT_COVER = os.path.join(DATA_DIR, "EMIT_L2A_RFL_001_20250405T172228_2509511_015__level1_pca_rf_refl_indices_spatial_cover.tif")
UAS_RGB = "/Volumes/drive/EMIT_project_CO/EMIT Field Campaign 052025/Sites/HT_05(05-06)2025/HT_RGB61_utm12n.tif"
UAS_CLASS = os.path.join(DATA_DIR, "HT_RF_points_level1_broad.tif")
CELL_FRACTIONS = os.path.join(DATA_DIR, "HT_level1_frac_60m.tif")
UAS_EXTENT = os.path.join(DATA_DIR, "HT_MS_extent.geojson")
UAS_CRS = "EPSG:26912"

# b: map crop and magnified spot of the original Fig. 6a
UAS_CROP_LONLAT = (-109.8355, -109.8295, 38.5794, 38.5885)   # lon W, lon E, lat S, lat N
ZOOM_CENTER_LONLAT = (-109.833, 38.58425)
ZOOM_RADIUS = 15.0           # m
MAP_PX = 3000                # longest side read for the UAS maps; the zoom is read at native resolution
EXAMPLE_CELL = (16, 6)       # (row, col) of the 60-m cell whose fractions are shown (close to the site mean)

# field photographs: (file, crop box as fractions of the upright image (x0, y0, x1, y1), label)
PHOTOS = [
    ("IMG_2543.jpg", (0.00, 0.155, 1.00, 0.905), "Sampling frame"),
    ("IMG_2549.JPG", (0.06, 0.20, 0.78, 0.74), "Lichen"),
    ("IMG_2550.JPG", (0.06, 0.34, 0.70, 0.82), "Moss"),
]

# UAS classes (codes in the classification raster) and colours shared by b and d
CLASSES = {
    1: ("Late successional biocrust", "#8c1d18"),
    2: ("Light surface", "#e8d9a8"),
    3: ("Rock", "#9ecae1"),
    4: ("Litter", "#2b59c3"),
    5: ("Vascular plants", "#33a02c"),
}

# d: components mapped; c: composite channels (R, G, B), in the order of the original Fig. 6b
COMPONENTS = ["Late successional biocrust", "Light surface", "Vascular plants"]
CHANNELS = ["Light surface", "Late successional biocrust", "Vascular plants"]
RAMP_ENDS = {"Late successional biocrust": "#8c1d18", "Light surface": "#b08a1e", "Vascular plants": "#2e7d32"}

MM = 1 / 25.4
FIG_W = 183 * MM
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "pdf.fonttype": 42,        # embed TrueType so text stays editable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
PANEL_LETTER_SIZE = 8
PANEL_TITLE_SIZE = 8
ARROW_COLOR = "#c62828"
ARROW_TEXT_COLOR = "#1f3a68"

# layout (inches)
SIDE = 0.04            # page margins
HEADING = 0.26         # panel letter + title above a row
GAP = 0.06             # gap between images inside a panel

HALO = [pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()]
TEXT_HALO = [pe.withStroke(linewidth=1.2, foreground="white")]


# ================================================================ helpers
def axes_in(fig, left, bottom, width, height, **kw):
    """Axes placed in inches from the figure's lower-left corner."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kw)


def heading(fig, x, y, letter, title):
    w, h = fig.get_size_inches()
    fig.text(x / w, y / h, letter, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="baseline")
    fig.text((x + 0.14) / w, y / h, title, fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="baseline")


def bare(ax, lw=0.6):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


def label_inside(ax, text):
    ax.text(0.03, 0.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, fontweight="bold",
            zorder=10, bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))


def scale_bar(ax, length, label, loc=(0.06, 0.04), at=None):
    """Scale bar; `length` in data units (metres, or degrees for lon/lat axes). Its left end is at `loc`
    (axes fraction) or at the data point `at`."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x, y = at if at is not None else (x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0))
    ax.plot([x, x + length], [y, y], color="black", lw=1.2, solid_capstyle="butt", path_effects=HALO, zorder=8)
    ax.text(x + length / 2, y + 0.012 * (y1 - y0), label, ha="center", va="bottom", fontsize=6,
            path_effects=TEXT_HALO, zorder=8)


def read_photo(fname, box, size=1400):
    im = ImageOps.exif_transpose(Image.open(os.path.join(DATA_DIR, fname))).convert("RGB")
    w, h = im.size
    im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    return np.asarray(im.resize((size, round(size * im.height / im.width)), Image.LANCZOS))


def read_raster(path, bands=None):
    """Array, [left, right, bottom, top] and band names."""
    with rasterio.open(path) as ds:
        arr = ds.read(bands) if bands else ds.read()
        b = ds.bounds
        return arr, [b.left, b.right, b.bottom, b.top], list(ds.descriptions)


def read_uas(path, bounds, bands, max_px=None, resampling=Resampling.nearest):
    """Window (left, bottom, right, top) of an original UAS raster, at native resolution or, with max_px,
    decimated (GDAL uses the file's overviews). Returns (array, [left, right, bottom, top])."""
    with rasterio.open(path) as ds:
        win = from_bounds(*bounds, ds.transform).round_offsets().round_lengths()
        shape = None
        if max_px:
            f = max(win.width, win.height) / max_px
            shape = (len(bands), round(win.height / f), round(win.width / f))
        arr = ds.read(bands, window=win, out_shape=shape, resampling=resampling, boundless=True, fill_value=0)
        left, bottom, right, top = window_bounds(win, ds.transform)
    return arr, [left, right, bottom, top]


def rgb_rgba(arr):
    """uint8 RGBA image from the orthomosaic's R, G, B and alpha bands; edge pixels that averaging mixed
    with the no-data fill are dropped."""
    img = arr.transpose(1, 2, 0).copy()
    img[..., 3] = np.where(img[..., 3] == 255, 255, 0)
    return img


def class_rgba(codes):
    """RGBA image of the classification; code 0 (unclassified/outside) transparent."""
    lut = np.zeros((256, 4), np.float32)
    for code, (_, color) in CLASSES.items():
        lut[code] = matplotlib.colors.to_rgba(color)
    return lut[codes]


def ramp(name):
    end = matplotlib.colors.to_rgb(RAMP_ENDS[name])
    return LinearSegmentedColormap.from_list(name, [(0.97, 0.97, 0.97), end, tuple(0.55 * np.array(end))])


# ================================================================ EMIT composite (method of 5_plot_figures.py)
def composite_rgb(arr, names, gain_pct=95, gain_target=0.17):
    """Three-component mixture -> colour. The predicted mixture varies by only a few percentage points, so
    deviations from the scene-mean mixture are multiplied by a gain before colouring (ratios preserved)."""
    comps = np.dstack([arr[names.index(n)] for n in CHANNELS]).astype("float64")
    valid = np.isfinite(comps).all(axis=-1)
    bary = comps / np.where(comps.sum(axis=-1, keepdims=True) < 1e-9, 1, comps.sum(axis=-1, keepdims=True))
    mean_b = np.array([np.nanmean(bary[..., i][valid]) for i in range(3)])
    dev = bary - mean_b
    gain = float(gain_target / np.percentile(np.abs(dev[valid]), gain_pct))
    enh = np.clip(mean_b + gain * dev, 0, 1)
    enh /= np.where(enh.sum(axis=-1, keepdims=True) < 1e-9, 1, enh.sum(axis=-1, keepdims=True))
    img = enh / np.where(enh.max(axis=-1, keepdims=True) < 1e-9, 1, enh.max(axis=-1, keepdims=True))
    return np.dstack([np.nan_to_num(np.clip(img, 0, 1), nan=1.0), valid]), gain, mean_b


def ternary_key(ax, labels, n=220):
    """Triangle whose colour at each mixture is the map colour for that mixture (R bottom-left, G top,
    B bottom-right)."""
    ax.set_aspect("equal")
    ax.axis("off")
    H = np.sqrt(3) / 2
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, H, n))
    g = ys / H
    b = xs - g / 2
    r = 1 - b - g
    inside = (r >= -1e-9) & (g >= -1e-9) & (b >= -1e-9)
    tri = np.dstack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)])
    tri = tri / np.maximum(tri.max(axis=-1, keepdims=True), 1e-9)
    ax.imshow(np.dstack([tri, inside]), origin="lower", extent=(0, 1, 0, H), interpolation="bilinear")
    ax.plot([0, 1, 0.5, 0], [0, 0, H, 0], color="#555555", lw=0.6)
    kw = dict(fontsize=6.5, linespacing=1.15, ha="center")
    ax.text(0.0, -0.06, labels[0], va="top", **kw)
    ax.text(0.5, H + 0.06, labels[1], va="bottom", **kw)
    ax.text(1.0, -0.06, labels[2], va="top", **kw)
    ax.set_xlim(-0.45, 1.3)
    ax.set_ylim(-0.42, H + 0.36)


def crop_valid(arr, extent):
    """Trim all-NaN border rows/cols; returns (arr, extent)."""
    v = np.isfinite(arr[0])
    r = np.where(v.any(axis=1))[0]
    c = np.where(v.any(axis=0))[0]
    h, w = v.shape
    x0, x1, y0, y1 = extent
    ext = [x0 + (x1 - x0) * c[0] / w, x0 + (x1 - x0) * (c[-1] + 1) / w,
           y1 - (y1 - y0) * (r[-1] + 1) / h, y1 - (y1 - y0) * r[0] / h]
    return arr[:, r[0]:r[-1] + 1, c[0]:c[-1] + 1], ext


def site_box(ax, min_km=2.5):
    """UAS footprint on a lon/lat map, enlarged to min_km so it stays visible at scene scale.
    Returns the box as (left, bottom, right, top) in degrees."""
    a = np.array(json.load(open(UAS_EXTENT))["features"][0]["geometry"]["coordinates"][0])
    cx, cy = (a[:, 0].min() + a[:, 0].max()) / 2, (a[:, 1].min() + a[:, 1].max()) / 2
    w = max(a[:, 0].max() - a[:, 0].min(), min_km / (111.32 * np.cos(np.deg2rad(cy))))
    h = max(a[:, 1].max() - a[:, 1].min(), min_km / 110.57)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="black", lw=0.8, path_effects=HALO,
                           zorder=9))
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


# ================================================================ panel b pieces
def loupe(ax, image, extent, center, radius, pos=(0.14, 0.64), width=0.60):
    """Circular magnified view of center +/- radius at pos (axes fraction, lower-left) in ax, with a gold
    ring at the spot and two leader lines, as in the original Fig. 6a."""
    fig = ax.figure
    bb = ax.get_position()
    W, H = fig.get_size_inches()
    height = width * bb.width * W / (bb.height * H)             # square on the page
    ins = ax.inset_axes((pos[0], pos[1], width, height))
    ins.set_zorder(8)
    im = ins.imshow(image, extent=extent, interpolation="antialiased")
    im.set_clip_path(Circle((0.5, 0.5), 0.5, transform=ins.transAxes))
    cx, cy = center
    ins.set_xlim(cx - radius, cx + radius)
    ins.set_ylim(cy - radius, cy + radius)
    ins.axis("off")
    ins.add_patch(Circle((0.5, 0.5), 0.5, transform=ins.transAxes, fill=False, ec="0.1", lw=1.0, zorder=10))
    ax.add_patch(Circle(center, radius, fill=False, ec="gold", lw=0.9, zorder=9))
    spot = ax.transData.transform(center)
    mid = ax.transAxes.transform((pos[0] + width / 2, pos[1] + height / 2))
    ang = np.arctan2(spot[1] - mid[1], spot[0] - mid[0])
    for da in (-0.5, 0.5):
        ax.add_artist(ConnectionPatch(xyA=center, coordsA="data", axesA=ax,
                                      xyB=(0.5 + 0.5 * np.cos(ang + da), 0.5 + 0.5 * np.sin(ang + da)),
                                      coordsB="axes fraction", axesB=ins, color="0.15", lw=0.5, zorder=7))


def footprint_blocks(codes, extent, block=12):
    """Block-averaged mask of the classified footprint (ragged specks along the edge drop out) and the
    block size in data units."""
    h, w = codes.shape
    hb, wb = h // block, w // block
    m = (codes[:hb * block, :wb * block] > 0).reshape(hb, block, wb, block).mean(axis=(1, 3)) > 0.5
    return m, block * (extent[1] - extent[0]) / w, block * (extent[3] - extent[2]) / h


def footprint_corners(codes, extent):
    """Top-right and bottom-right corners (data units) of the classified footprint: the extreme blocks
    along the two diagonals."""
    m, dx, dy = footprint_blocks(codes, extent)
    rows, cols = np.nonzero(m)
    xs = extent[0] + (cols + 1) * dx                     # right edge of each block
    top, bottom = extent[3] - rows * dy, extent[3] - (rows + 1) * dy
    i, j = np.argmax(xs + top), np.argmax(xs - bottom)
    return (xs[i], top[i]), (xs[j], bottom[j])


def footprint_right_edge(codes, extent, y):
    """Point (data units) on the right edge of the classified footprint at height y."""
    m, dx, dy = footprint_blocks(codes, extent)
    row = int((extent[3] - y) / dy)
    return extent[0] + (np.nonzero(m[row])[0].max() + 1) * dx, y


def cell_columns(ax, fractions):
    """Class fractions (%) of one 60-m cell as columns in class colours (named by the class legend)."""
    colors = [CLASSES[c][1] for c in CLASSES]
    x = np.arange(len(colors))
    ax.bar(x, [100 * f for f in fractions], color=colors, edgecolor="#555555", linewidth=0.4, width=0.42)
    for xi, f in zip(x, fractions):
        ax.text(xi, 100 * f + 1.5, f"{100 * f:.0f}", ha="center", va="bottom", fontsize=6)
    ax.set_xlim(-0.6, len(colors) - 0.4)
    ax.set_ylim(0, 45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Cover (%)", fontsize=6, labelpad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)


def class_legend(ax):
    handles = [Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=name)
               for name, color in CLASSES.values()]
    leg = ax.legend(handles=handles, loc="center left", ncol=1, frameon=False, fontsize=6.5, handlelength=1.1,
                    handleheight=1.0, handletextpad=0.5, labelspacing=0.75, borderpad=0, borderaxespad=0,
                    title="UAS classes", title_fontproperties={"size": 6.5, "weight": "bold"})
    leg._legend_box.align = "left"
    ax.axis("off")


def to_inches(ax, xy):
    """Figure position (inches from the lower-left corner) of a data point of ax."""
    ax.apply_aspect()
    return ax.transData.transform(xy) / ax.figure.dpi


def upscale_arrows(fig, ax_from, starts, ax_to, box, label):
    """Two dashed arrows from the points `starts` (data units of ax_from; upper first) to the left corners of
    box (data units of ax_to), with `label` written along the upper arrow."""
    ax_to.apply_aspect()
    x0, y0, x1, y1 = box
    for start, target in zip(starts, ((x0, y1), (x0, y0))):
        fig.add_artist(ConnectionPatch(xyA=start, coordsA="data", axesA=ax_from, xyB=target,
                                       coordsB="data", axesB=ax_to, arrowstyle="-|>", mutation_scale=6,
                                       color=ARROW_COLOR, lw=0.8, linestyle=(0, (3, 2)), shrinkA=0,
                                       shrinkB=1, zorder=12))
    W, H = fig.get_size_inches()
    p0 = ax_from.transData.transform(starts[0]) / fig.dpi                   # inches
    p1 = ax_to.transData.transform((x0, y1)) / fig.dpi
    ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    p = p0 + 0.45 * (p1 - p0) + 0.025 * np.array([np.sin(ang), -np.cos(ang)])   # just below the line
    fig.text(p[0] / W, p[1] / H, label, rotation=np.degrees(ang), rotation_mode="anchor", ha="center",
             va="top", fontsize=6.5, fontweight="bold", fontstyle="italic", color=ARROW_TEXT_COLOR, zorder=12)


# ================================================================ figure
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(NATURE_RC)
    if not os.path.exists(UAS_RGB):
        raise SystemExit(f"UAS orthomosaic not found (is the external drive mounted?): {UAS_RGB}")

    # ---------- data
    photos = [read_photo(f, box) for f, box, _ in PHOTOS]
    to_utm = Transformer.from_crs("EPSG:4326", UAS_CRS, always_xy=True)
    lw_, le_, ls_, ln_ = UAS_CROP_LONLAT
    xs, ys = to_utm.transform([lw_, le_, le_, lw_], [ln_, ln_, ls_, ls_])
    crop = (min(xs), min(ys), max(xs), max(ys))
    zc = to_utm.transform(*ZOOM_CENTER_LONLAT)
    zoom = (zc[0] - ZOOM_RADIUS, zc[1] - ZOOM_RADIUS, zc[0] + ZOOM_RADIUS, zc[1] + ZOOM_RADIUS)
    rgb, ext_rgb = read_uas(UAS_RGB, crop, [1, 2, 3, 4], MAP_PX, Resampling.average)
    rgb_zoom, ext_rgb_zoom = read_uas(UAS_RGB, zoom, [1, 2, 3, 4])
    codes, ext_cls = read_uas(UAS_CLASS, crop, [1], MAP_PX)
    codes_zoom, ext_cls_zoom = read_uas(UAS_CLASS, zoom, [1])
    with rasterio.open(CELL_FRACTIONS) as ds:
        frac = ds.read()
        grid = ds.bounds
        cell_xy = ds.xy(*EXAMPLE_CELL)
        frac_names = list(ds.descriptions)
    cell_frac = [frac[frac_names.index(CLASSES[c][0])][EXAMPLE_CELL] for c in CLASSES]
    cell_box = (cell_xy[0] - 30, cell_xy[1] - 30, cell_xy[0] + 30, cell_xy[1] + 30)
    codes_cell, ext_cell = read_uas(UAS_CLASS, cell_box, [1])
    emit, ext_emit, names = read_raster(EMIT_COVER)
    emit, ext_emit = crop_valid(emit.astype("float64"), ext_emit)
    composite, gain, mean_b = composite_rgb(emit, names)
    print(f"composite gain {gain:.1f}x, scene-mean mixture " +
          ", ".join(f"{n} {m:.2f}" for n, m in zip(CHANNELS, mean_b)))
    print("example cell", EXAMPLE_CELL, ", ".join(f"{CLASSES[c][0]} {100 * f:.0f}%" for c, f in
                                                 zip(CLASSES, cell_frac)))

    uas_aspect = (crop[2] - crop[0]) / (crop[3] - crop[1])
    lat0 = np.mean(ext_emit[2:])
    emit_aspect = (ext_emit[1] - ext_emit[0]) * np.cos(np.deg2rad(lat0)) / (ext_emit[3] - ext_emit[2])

    # ---------- layout: left column a over b | c composite, with the one-pixel example (right of b), class
    # legend and ternary key below it | d three maps stacked
    left_w = 2.6
    s = (left_w - 0.5 * GAP) / 1.5             # large photo; two small ones stacked beside it
    small = (s - GAP) / 2
    uas_w = (left_w - GAP) / 2
    uas_h = uas_w / uas_aspect
    label_h = 0.13                             # "RGB" / "Classification" above the maps
    H = HEADING + s + HEADING + label_h + uas_h + 0.04
    fig = plt.figure(figsize=(FIG_W, H))
    W = FIG_W
    top = H - HEADING

    # a: photographs
    heading(fig, SIDE, top + 0.08, "a", "Biocrusts in the field")
    spots = [(SIDE, top - s, s), (SIDE + s + GAP, top - small, small), (SIDE + s + GAP, top - s, small)]
    for (x, y, size), img, (_, _, label) in zip(spots, photos, PHOTOS):
        ax = axes_in(fig, x, y, size, size)
        ax.imshow(img)
        bare(ax)
        label_inside(ax, label)

    # b: UAS RGB | classification with the 60-m EMIT grid, magnified detail
    b_top = top - s - HEADING
    maps_top = b_top - label_h
    b_bottom = maps_top - uas_h
    heading(fig, SIDE, b_top + 0.08, "b", "UAS imagery and classification")
    cls_left = SIDE + uas_w + GAP
    for x, text in ((SIDE, "RGB"), (cls_left, "Classification")):
        fig.text(x / W, (maps_top + 0.035) / H, text, fontsize=6.5, fontweight="bold", va="baseline")
    ax_rgb = axes_in(fig, SIDE, b_bottom, uas_w, uas_h)
    ax_cls = axes_in(fig, cls_left, b_bottom, uas_w, uas_h)
    ax_rgb.imshow(rgb_rgba(rgb), extent=ext_rgb, interpolation="antialiased")
    ax_cls.imshow(class_rgba(codes[0]), extent=ext_cls, interpolation="nearest")
    for ax in (ax_rgb, ax_cls):
        ax.set_xlim(crop[0], crop[2])
        ax.set_ylim(crop[1], crop[3])
        ax.axis("off")
    scale_bar(ax_rgb, 200, "200 m", loc=(0.05, 0.03))
    for x in np.arange(grid.left, grid.right + 1, 60):
        ax_cls.axvline(x, color="white", lw=0.25, alpha=0.8, zorder=3)
    for y in np.arange(grid.bottom, grid.top + 1, 60):
        ax_cls.axhline(y, color="white", lw=0.25, alpha=0.8, zorder=3)
    ax_cls.add_patch(Rectangle(cell_box[:2], 60, 60, fill=False, ec="black", lw=0.8, path_effects=HALO,
                               zorder=9))
    loupe(ax_rgb, rgb_rgba(rgb_zoom), ext_rgb_zoom, zc, ZOOM_RADIUS)
    loupe(ax_cls, class_rgba(codes_zoom[0]), ext_cls_zoom, zc, ZOOM_RADIUS)

    # c: three-component composite; arrows from the classification's corners to the site in the scene
    c_left = SIDE + left_w + 0.26
    d_w = 1.36
    d_left = FIG_W - SIDE - d_w
    comp_w = d_left - 0.08 - c_left
    comp_h = comp_w / emit_aspect
    heading(fig, c_left, top + 0.08, "c", "Upscaling to EMIT hyperspectral imagery")
    ax_comp = axes_in(fig, c_left, top - comp_h, comp_w, comp_h)
    ax_comp.imshow(composite, extent=ext_emit, interpolation="nearest")
    ax_comp.set_aspect(1 / np.cos(np.deg2rad(lat0)))
    ax_comp.axis("off")
    box = site_box(ax_comp)
    ax_comp.text(box[2] + 0.004, box[1] - 0.004, "UAS training area", ha="left", va="top", fontsize=6.5,
                 fontweight="bold", fontstyle="italic", color=ARROW_TEXT_COLOR, path_effects=TEXT_HALO,
                 zorder=12)
    # upper arrow from the top-right corner; lower one from the right edge one cell above the example cell,
    # so it clears the example's connector
    starts = (footprint_corners(codes[0], ext_cls)[0], footprint_right_edge(codes[0], ext_cls, cell_box[3] + 60))
    upscale_arrows(fig, ax_cls, starts, ax_comp, box, "Upscale to EMIT scale")

    # under the composite: class legend (b, d) and ternary key (c) side by side, and below them, right of b,
    # one 60-m EMIT pixel (classified cell and its class fractions), kept just right of the lower arrow
    p0, p1 = to_inches(ax_cls, starts[1]), to_inches(ax_comp, box[:2])
    arrow_x = lambda y: p0[0] + (y - p0[1]) * (p1[0] - p0[0]) / (p1[1] - p0[1])
    right = d_left - 0.08
    row_bottom, row_h = 0.98, 0.78             # legend + key row, top kept below the composite's lower tip
    key_w = row_h * 1.75 / 1.646               # ternary_key's data limits are 1.75 x 1.646
    ax_key = axes_in(fig, right - key_w - 0.06, row_bottom, key_w, row_h)
    ternary_key(ax_key, ["Light\nsurface", "Late successional\nbiocrust", "Vascular\nplants"])
    # composite scale bar, centred above the key (in the empty corner below the scene)
    bar = 20 / (111.32 * np.cos(np.deg2rad(lat0)))
    key_mid = right - key_w - 0.06 + key_w * 0.95 / 1.75      # triangle's centre line, inches
    bx, by = ax_comp.transData.inverted().transform(np.array([key_mid, row_bottom + row_h + 0.28]) * fig.dpi)
    scale_bar(ax_comp, bar, "20 km", at=(bx - bar / 2, by))
    class_legend(axes_in(fig, right - key_w - 1.32, row_bottom, 1.3, row_h))     # centred on the key

    cell_side, ex_bottom = 0.68, 0.08
    ex_left = arrow_x(ex_bottom + cell_side + 0.2) + 0.06
    ax_cell = axes_in(fig, ex_left, ex_bottom, cell_side, cell_side)
    ax_cell.imshow(class_rgba(codes_cell[0]), extent=ext_cell, interpolation="antialiased")
    bare(ax_cell, lw=0.8)
    fig.text(ex_left / W, (ex_bottom + cell_side + 0.06) / H, "One 60-m EMIT pixel", fontsize=6.5,
             fontweight="bold", va="baseline")
    cols_left = ex_left + cell_side + 0.18
    cell_columns(axes_in(fig, cols_left, ex_bottom, right - 0.1 - cols_left, cell_side - 0.08), cell_frac)
    ax_cls.add_artist(ConnectionPatch(xyA=(cell_box[2], cell_xy[1]), coordsA="data", axesA=ax_cls,
                                      xyB=(0, 0.5), coordsB="axes fraction", axesB=ax_cell, color="black",
                                      lw=0.6, linestyle=(0, (2, 1.5)), zorder=7))

    # d: predicted fractional cover, maps stacked with a colour bar under each
    heading(fig, d_left, top + 0.08, "d", "Fractional cover")
    block = (top - 0.04) / 3
    map_h = min(block - 0.40, (d_w - 0.04) / emit_aspect)
    map_w = map_h * emit_aspect
    for k, name in enumerate(COMPONENTS):
        y = top - k * block - map_h
        ax = axes_in(fig, d_left + (d_w - map_w) / 2, y, map_w, map_h)
        band = 100 * emit[names.index(name)]
        lo, hi = np.nanpercentile(band, (2, 98))
        im = ax.imshow(np.ma.masked_invalid(band), extent=ext_emit, cmap=ramp(name), vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        ax.axis("off")
        site_box(ax, min_km=4)
        cax = axes_in(fig, d_left + 0.14, y - 0.1, d_w - 0.28, 0.055)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1)
        cb.locator = MaxNLocator(nbins=3)
        cb.update_ticks()
        cb.set_label(f"{name} (%)", fontsize=6, labelpad=1.5)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"Fig6.{ext}")
        fig.savefig(out, dpi=1000, facecolor="white")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
