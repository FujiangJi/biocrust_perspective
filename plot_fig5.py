"""Figure 5: spatial enhancement of EMIT observations and derived spectral-index patterns.

Same arrangement as the earlier Fig. 5:
a  study area within the Colorado Plateau (red square = example scene)
b  RGB composite (R 671, G 552, B 455 nm): EMIT 60 m | fused 4 m (EMIT + PlanetScope)
c  spatial sampling of one 180 x 180 m block: Google Maps imagery, EMIT 60-m pixels, PlanetScope 4-m pixels
d-g  CI, SCI, BOMI, SCAI: EMIT 60 m | fused 4 m, one shared colour range per index (colour bar inside the
     fused map)

Each map pair has the same extent and grid (EMIT cropped to the fused footprint), a scale bar, and the same
zoom box (small box) linked to a magnified view of it (inset, upper right), so fine-scale detail can be
compared at one location.

Sized for Nature: 183 mm wide (double column), Arial, 1000 dpi PNG plus vector PDF.

Run with the Fujiang env:
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_fig5.py
"""
import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/0_src_code/1_data"
FIG5_DIR = os.path.join(DATA_DIR, "5_Fig5_data")
FIG_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/figures"

EMIT_TIF = os.path.join(FIG5_DIR, "1_train_EMIT_resampled_INDICES.tif")
FUSED_TIF = os.path.join(FIG5_DIR, "1_train_MSAHFNET_fused_full_spectrum_INDICES.tif")
CO_PLATEAU = os.path.join(DATA_DIR, "CO_Plateau_wgs84.geojson")
SAMPLING_PNGS = {"imagery": "1.png", "emit": "2.png", "fused": "3.png"}   # panel c tiles (same 180 m block)
SRC_EPSG = 32612

EMIT_LABEL = "EMIT (60 m)"
FUSED_LABEL = "Fused imagery (4 m)"
RGB_BANDS = ("671 nm", "552 nm", "455 nm")

# (panel letter, band name in the GeoTIFFs, short name, full name); BOCI is shown as BOMI
INDEX_PANELS = [
    ("d", "CI", "CI", "cyanobacterial crust index"),
    ("e", "SCI", "SCI", "scytonemin index"),
    ("f", "BOCI", "BOMI", "biocrust organic matter index"),
    ("g", "SCAI", "SCAI", "satellite cellulose absorption index"),
]

# zoom box shown in b and d-g (UTM 12N metres): escarpment, slope and road west of the cliff
ZOOM = (404300, 4071100, 406300, 4072665)   # left, bottom, right, top (same aspect as the scene)
ZOOM_INSET = (0.60, 0.57, 0.38, 0.38)       # inset position in each map (axes fraction)

MM = 1 / 25.4
FIG_W = 183 * MM
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.cal": "Arial",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,        # embed TrueType so text stays editable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
PANEL_LETTER_SIZE = 8
PANEL_TITLE_SIZE = 8

# layout (inches)
SIDE = 0.04            # right page margin
LEFT = 0.04            # left page margin (left edge of a's labels and of the d/f boxes)
LAT_W = 0.13           # room for panel a's rotated latitude labels, left of its map
C_COL_GAP = 0.32       # gap between panel c's two tile columns
BOX_GAP = 0.14         # gap between the two pair boxes in a row
PAD = 0.05             # padding inside a pair box
MAP_GAP = 0.06         # gap between the two maps of a pair
HEADING = 0.26         # panel letter + title above a box

HALO = [pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()]
TEXT_HALO = [pe.withStroke(linewidth=1.2, foreground="white")]


# ================================================================ data
def read_bands(path, names, bounds=None):
    """Named bands as float arrays (nodata -> NaN), optionally cropped to bounds; plus [l, r, b, t]."""
    with rasterio.open(path) as ds:
        window = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths() if bounds else None
        out = {}
        for name in names:
            arr = ds.read(ds.descriptions.index(name) + 1, window=window).astype(np.float32)
            arr[(arr == ds.nodata) | ~np.isfinite(arr)] = np.nan
            out[name] = arr
        transform = ds.window_transform(window) if window is not None else ds.transform
        h, w = out[names[0]].shape
        left, bottom, right, top = rasterio.transform.array_bounds(h, w, transform)
    return out, [left, right, bottom, top]


def shared_iqr(a, b, k=1.5):
    """Tukey limits (Q1 - k IQR, Q3 + k IQR) of the pooled values of two images."""
    pooled = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    q1, q3 = np.percentile(pooled, (25, 75))
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def make_rgb(bands, limits, gamma=0.7):
    """RGBA image; each band stretched with limits shared by EMIT and fused."""
    chans = [np.clip((bands[n] - limits[n][0]) / (limits[n][1] - limits[n][0]), 0, 1) ** gamma for n in RGB_BANDS]
    valid = np.all([np.isfinite(bands[n]) for n in RGB_BANDS], axis=0)
    return np.dstack([np.nan_to_num(np.dstack(chans)), valid]).astype(np.float32)


# ================================================================ drawing helpers
def axes_in(fig, left, bottom, width, height, **kw):
    """Axes placed in inches from the figure's lower-left corner."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kw)


def heading(fig, x, y, letter, title):
    """Bold panel letter and title, left-aligned at (x, y) inches (baseline)."""
    w, h = fig.get_size_inches()
    t = fig.text(x / w, y / h, letter, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="baseline")
    fig.text((x + 0.14) / w, y / h, title, fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="baseline")
    return t


def pair_box(fig, left, bottom, width, height):
    w, h = fig.get_size_inches()
    fig.add_artist(Rectangle((left / w, bottom / h), width / w, height / h, transform=fig.transFigure,
                             fill=False, edgecolor="#333333", lw=0.8))


def scale_bar(ax, km, loc=(0.05, 0.05)):
    """Horizontal scale bar in metre axes."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x, y = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + km * 1000], [y, y], color="black", lw=1.2, solid_capstyle="butt", path_effects=HALO, zorder=6)
    ax.text(x + km * 500, y + 0.02 * (y1 - y0), f"{km} km", ha="center", va="bottom", fontsize=6.5,
            path_effects=TEXT_HALO, zorder=6)


def zoom_box_and_inset(ax, image, extent, **imshow_kw):
    """Outline the zoom window (small box) and show it magnified in an inset, joined by two lines."""
    l, b, r, t = ZOOM
    line_halo = [pe.Stroke(linewidth=1.5, foreground="white"), pe.Normal()]
    ax.plot([l, r, r, l, l], [b, b, t, t, b], color="black", lw=0.7, path_effects=line_halo, zorder=7)
    inset = ax.inset_axes(ZOOM_INSET)
    inset.set_zorder(8)
    inset.imshow(image, extent=extent, origin="upper", interpolation="nearest", **imshow_kw)
    inset.set_xlim(l, r)
    inset.set_ylim(b, t)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)
    for box_corner, inset_corner in (((r, t), (0, 1)), ((r, b), (0, 0))):
        ax.add_artist(ConnectionPatch(xyA=box_corner, coordsA="data", axesA=ax, xyB=inset_corner,
                                      coordsB="axes fraction", axesB=inset, color="black", lw=0.5,
                                      path_effects=line_halo, zorder=7))


def inside_colorbar(ax, im, pad=0.015, right=0.985, bottom=0.04, height=0.4):
    """Vertical colour bar in the map's lower-right corner, on a white background fitted to it."""
    renderer = ax.figure.canvas.get_renderer()

    def place(x):
        cax = ax.inset_axes((x, bottom + pad, 0.028, height))
        cax.set_zorder(9)
        cb = plt.colorbar(im, cax=cax, extend="both", extendfrac=0.06)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6.5, width=0.5, length=2, pad=1.5)
        cb.locator = MaxNLocator(nbins=3)
        cb.update_ticks()
        return cax, cax.get_tightbbox(renderer).transformed(ax.transAxes.inverted())

    cax, bb = place(0.8)                       # measure, then shift so the tick labels end at `right`
    cax.remove()
    cax, bb = place(0.8 + right - pad - bb.x1)
    ax.add_patch(Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
                           transform=ax.transAxes, facecolor="white", edgecolor="none", alpha=0.9, zorder=8))


def show_map(ax, image, extent, km=2, **imshow_kw):
    """Scene map in UTM metres with a scale bar and the zoom box/inset; no ticks."""
    im = ax.imshow(image, extent=extent, origin="upper", interpolation="nearest", **imshow_kw)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    scale_bar(ax, km)
    zoom_box_and_inset(ax, image, extent, **imshow_kw)
    return im


def draw_pair(fig, left, top, map_w, map_h, images, extents, title, letter, cbar=False, **imshow_kw):
    """Boxed EMIT | fused pair, sensor label inside each map; optional colour bar inside the fused map."""
    w_box = 2 * map_w + MAP_GAP + 2 * PAD
    h_box = PAD + map_h + PAD
    bottom = top - h_box
    pair_box(fig, left, bottom, w_box, h_box)
    heading(fig, left, top + 0.08, letter, title)
    map_y = top - PAD - map_h
    im = None
    for k, (image, extent, label) in enumerate(zip(images, extents, (EMIT_LABEL, FUSED_LABEL))):
        x = left + PAD + k * (map_w + MAP_GAP)
        ax = axes_in(fig, x, map_y, map_w, map_h)
        im = show_map(ax, image, extent, **imshow_kw)
        ax.text(0.025, 0.975, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                fontweight="bold", linespacing=1.15, zorder=10,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    if cbar:
        inside_colorbar(ax, im)              # ax: the fused map (drawn last)
    return w_box, h_box


def draw_study_area(fig, left, bottom, width, height, scene_extent):
    """Colorado Plateau and surrounding states; red square marks the example scene."""
    ax = axes_in(fig, left, bottom, width, height, projection=ccrs.PlateCarree())
    # true local shape: 1 deg of latitude drawn 1/cos(lat) times 1 deg of longitude; the longitude range
    # is widened so the map fills the panel instead of leaving it narrow
    lat0, lat1, lon_c = 33.0, 41.6, -109.7
    stretch = 1 / np.cos(np.deg2rad((lat0 + lat1) / 2))
    half = width / height * (lat1 - lat0) * stretch / 2
    ax.set_extent([lon_c - half, lon_c + half, lat0, lat1], crs=ccrs.PlateCarree())
    ax.set_aspect(stretch)
    ax.set_anchor("NW")                        # map frame starts at the axes' left edge
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#fafafa", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#cfe6f4", edgecolor="none")
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="black", linewidth=0.5, facecolor="none", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black", linewidth=0.7, zorder=4)
    ax.add_geometries(gpd.read_file(CO_PLATEAU).geometry, crs=ccrs.PlateCarree(), facecolor="#e8a96c",
                      edgecolor="#a86a32", linewidth=0.4, zorder=2)
    to_geo = Transformer.from_crs(SRC_EPSG, 4326, always_xy=True)
    lon, lat = to_geo.transform((scene_extent[0] + scene_extent[1]) / 2, (scene_extent[2] + scene_extent[3]) / 2)
    ax.plot(lon, lat, marker="s", ms=6, mfc="none", mec="red", mew=1.4, transform=ccrs.PlateCarree(), zorder=6)
    for name, (lx, ly) in {"Utah": (-113.3, 40.4), "Colorado": (-105.7, 39.6),
                           "Arizona": (-113.4, 33.55), "New Mexico": (-106.2, 33.55)}.items():
        ax.text(lx, ly, name, fontsize=6.5, ha="center", va="center", transform=ccrs.PlateCarree(), zorder=6,
                path_effects=TEXT_HALO)
    grd = ax.gridlines(draw_labels=True, xlocs=range(-114, -101, 4), ylocs=range(34, 42, 3), color="gray",
                       linestyle="--", linewidth=0.4, x_inline=False, y_inline=False, zorder=5)
    grd.xpadding = grd.ypadding = 2            # points; keeps the latitude labels within LAT_W
    grd.top_labels = False
    grd.right_labels = False
    grd.xlabel_style = {"size": 5.5, "rotation": 0}
    grd.ylabel_style = {"size": 5.5, "rotation": 90, "va": "center"}
    # north arrow in the empty top-left corner (Nevada), fully inside the frame
    ax.annotate("", xy=(0.06, 0.87), xytext=(0.06, 0.76), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=7))
    ax.text(0.06, 0.88, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=7, fontweight="bold",
            path_effects=TEXT_HALO)
    ax.spines["geo"].set_linewidth(0.8)
    return ax


def draw_sampling(fig, left, bottom, width, height):
    """Panel c: one 180 x 180 m block as Google Maps imagery (60-m grid, 4-m grid in one cell),
    EMIT 60-m pixels and PlanetScope 4-m pixels; all three tiles the same size."""
    W, H = fig.get_size_inches()
    tiles = {k: plt.imread(os.path.join(FIG5_DIR, f)) for k, f in SAMPLING_PNGS.items()}
    s = (height - 0.03) / 2                    # one size for all three tiles (two stacked fill the height)
    img_x = left + 0.14                        # room for the "60 m" bracket
    img_y = bottom + (height - s) / 2
    px_x = img_x + s + C_COL_GAP               # the gap holds the "4-m grid" label

    def tile(key, x, y, label, above):
        ax = axes_in(fig, x, y, s, s)
        ax.imshow(tiles[key], extent=[0, 1, 0, 1], interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.text(0.5, 1.03 if above else -0.03, label, transform=ax.transAxes, ha="center",
                va="bottom" if above else "top", fontsize=6.5, fontweight="bold")
        return ax

    # labels sit in free space: above the top tile (beside the "c" title), below the bottom tile
    ax_img = tile("imagery", img_x, img_y, "Google Maps", above=False)
    ax_emit = tile("emit", px_x, bottom + height - s, "EMIT pixels", above=True)
    ax_fused = tile("fused", px_x, bottom, "PlanetScope pixels", above=False)

    # 60 m: one cell of the 3 x 3 grid (left side); 4 m: the gridded top-right cell
    ax_img.annotate("", xy=(-0.06, 2 / 3), xytext=(-0.06, 1.0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="|-|", lw=0.6, mutation_scale=2.5, shrinkA=0, shrinkB=0))
    ax_img.text(-0.12, 5 / 6, "60 m", transform=ax_img.transAxes, rotation=90, ha="center", va="center",
                fontsize=6)
    ax_img.annotate("", xy=(1.04, 2 / 3), xytext=(1.04, 1.0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", lw=0.6, shrinkA=0, shrinkB=0))
    ax_img.text(1.08, 5 / 6, "4-m\ngrid", transform=ax_img.transAxes, ha="left", va="center", fontsize=6,
                linespacing=1.1)
    ax_emit.text(0.5, 5 / 6, "60 m", transform=ax_emit.transAxes, ha="center", va="center", fontsize=6)
    ax_fused.text(0.97, 0.04, "4 m", transform=ax_fused.transAxes, ha="right", va="bottom", fontsize=6,
                  path_effects=TEXT_HALO)


# ================================================================ figure
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(NATURE_RC)

    names = [b for _, b, _, _ in INDEX_PANELS] + list(RGB_BANDS)
    fused, ext_f = read_bands(FUSED_TIF, names)
    emit, ext_e = read_bands(EMIT_TIF, names, bounds=(ext_f[0], ext_f[2], ext_f[1], ext_f[3]))
    print(f"EMIT {emit['CI'].shape} px, fused {fused['CI'].shape} px, extent {ext_f}")

    # shared display limits for each EMIT | fused pair
    rgb_limits = {n: np.percentile(np.concatenate([emit[n][np.isfinite(emit[n])], fused[n][np.isfinite(fused[n])]]),
                                   (2, 98)) for n in RGB_BANDS}
    index_limits = {b: shared_iqr(emit[b], fused[b]) for _, b, _, _ in INDEX_PANELS}
    for b, (lo, hi) in index_limits.items():
        print(f"  {b}: shared colour range {lo:.3f} - {hi:.3f}")

    # map size from the pair-box width (two boxes per row across the page)
    aspect = (ext_f[1] - ext_f[0]) / (ext_f[3] - ext_f[2])
    box_w = (FIG_W - LEFT - SIDE - BOX_GAP) / 2
    map_w = (box_w - 2 * PAD - MAP_GAP) / 2
    map_h = map_w / aspect
    row1_h = row_h = PAD + map_h + PAD
    H = 3 * HEADING + row1_h + 2 * row_h + 0.05
    fig = plt.figure(figsize=(FIG_W, H))

    # row 1: a study area | b RGB pair | c spatial sampling
    top = H - HEADING
    tile_s = (row1_h - 0.03) / 2
    c_w = 0.14 + 2 * tile_s + C_COL_GAP + 0.06                     # panel c: bracket, two tile columns, label overhang
    a_w = FIG_W - LEFT - SIDE - box_w - c_w - 0.12 - 0.08
    b_left = LEFT + a_w + 0.12
    draw_study_area(fig, LEFT + LAT_W, top - row1_h + 0.12, a_w - LAT_W, row1_h - 0.12, ext_f)
    heading(fig, LEFT, top + 0.08, "a", "Study area")
    draw_pair(fig, b_left, top, map_w, map_h, [make_rgb(emit, rgb_limits), make_rgb(fused, rgb_limits)],
              [ext_e, ext_f], "RGB composite (R 671, G 552, B 455 nm)", "b")
    c_left = b_left + box_w + 0.08
    heading(fig, c_left, top + 0.08, "c", "Spatial sampling")
    draw_sampling(fig, c_left, top - row1_h, FIG_W - SIDE - c_left, row1_h)

    # rows 2-3: index pairs d-g
    for k, (letter, band, short, name) in enumerate(INDEX_PANELS):
        row, col = divmod(k, 2)
        top = H - HEADING - row1_h - HEADING - row * (row_h + HEADING)
        left = LEFT + col * (box_w + BOX_GAP)
        lo, hi = index_limits[band]
        draw_pair(fig, left, top, map_w, map_h, [emit[band], fused[band]], [ext_e, ext_f],
                  f"{short} ({name})", letter, cbar=True, cmap="viridis", vmin=lo, vmax=hi)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"Fig5.{ext}")
        fig.savefig(out, dpi=1000, facecolor="white")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
