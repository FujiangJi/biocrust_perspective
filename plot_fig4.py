"""Figure 4: multiscale view of biocrust-sensitive spectral indices (PACE -> EMIT -> AVIRIS-5).

Layout: a across the top; b and c side by side below it, with the legend column to their right.
a  PACE OCI, full globe with drylands coloured (monthly composite, May 2025); box "b" = Colorado Plateau
b  EMIT, Colorado Plateau (Oct 2023 - Dec 2025 mosaic); outline "c" = AVIRIS-5 flight-line coverage
c  AVIRIS-5, all flight lines (single day, 8 May 2025)

All three panels use one four-index colour composite (CI, SCI, SCAI, BOMI) with the same fixed index
ranges and the same rectangle colour key. Each pixel's colour is the key colour at the pixel's position
in the key, where the position comes from the relative balance of its four (range-scaled) indices.
A second figure (Fig4_SI_single_indices) shows each index on its own for the SI.

Sized for Nature: 183 mm wide (double column), Arial 5-8 pt, 1000 dpi PNG plus vector PDF.

Run with the Fujiang env (rasterio, cartopy, geopandas, pyproj, scipy):
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_fig4.py
"""
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.transform import from_bounds
import geopandas as gpd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

DATA_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/0_src_code/1_data"
FIG4_DIR = os.path.join(DATA_DIR, "4_Fig4_data")
FIG_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/figures"

PACE_TIF = os.path.join(FIG4_DIR, "group_2025_05.tif")
EMIT_TIF = os.path.join(FIG4_DIR, "all_data_mosaiced.tif")
AVIRIS_TIF = os.path.join(FIG4_DIR, "all_20250508.tif")
ARIDITY_TIF = os.path.join(DATA_DIR, "TERRACLIMATE_ARIDITY_GLOBAL_GEO_4km_classes_noice.tif")
CO_PLATEAU = os.path.join(DATA_DIR, "CO_Plateau_wgs84.geojson")

# reading the full-resolution GeoTIFFs is slow (PACE has no overviews), so the display-resolution
# arrays are cached here; delete the file (or set REBUILD_CACHE) after the GeoTIFFs change
CACHE = os.path.join(FIG4_DIR, "fig4_display_cache.npz")
REBUILD_CACHE = False

INDICES = ["CI", "SCI", "SCAI", "BOCI"]   # band names in the GeoTIFFs (BOCI is shown as BOMI)

PACE_DECIMATE = 10      # 0.01 deg -> 0.1 deg
EMIT_DECIMATE = 8       # ~60 m -> ~480 m
AVIRIS_DECIMATE = 16    # 10.6 m -> ~170 m
DRYLAND_CLASSES = (1, 2, 3, 4)   # UNEP aridity classes with AI < 0.65 (dry sub-humid to hyper-arid)

# ---------------------------------------------------------------- colour encoding
# One fixed value range per index, shared by every panel: pooled 2nd-98th percentile of the three
# platforms (equal weight per platform), rounded. Values outside the range are clipped.
INDEX_RANGES = {"CI": (0.40, 1.00), "SCI": (0.00, 0.55), "SCAI": (0.00, 0.32), "BOCI": (0.07, 0.24)}
INDEX_LABELS = {"CI": "CI", "SCI": "SCI", "SCAI": "SCAI", "BOCI": "BOMI"}
INDEX_NAMES = {"CI": "Cyanobacterial crust index", "SCI": "Scytonemin index",
               "SCAI": "Satellite cellulose absorption index", "BOCI": "Biocrust organic matter index"}

# Rectangle key: index at each corner and its colour. x runs left -> right, y bottom -> top.
CORNERS = {             # (x, y), RGB
    "CI":   ((0, 1), (1.0, 0.0, 0.0)),
    "SCI":  ((1, 1), (0.0, 1.0, 0.0)),
    "BOCI": ((1, 0), (0.0, 0.0, 1.0)),
    "SCAI": ((0, 0), (1.0, 1.0, 0.0)),
}
KEY_ZOOM_PCT = (1, 99)   # the key spans these percentiles of pixel positions, pooled over the three panels

# Platform colours mark each panel title and the platform table in the legend strip; match them to the
# colours of the platform table in the manuscript
PLATFORMS = {
    "PACE":   dict(label="PACE OCI", color="#5e3c99", region="global drylands", grid="0.01° (~1 km)",
                   date="May 2025", kind="monthly composite"),
    "EMIT":   dict(label="EMIT", color="#d01c8b", region="Colorado Plateau", grid="~60 m",
                   date="Oct 2023–Dec 2025", kind="multi-year mosaic"),
    "AVIRIS": dict(label="AVIRIS-5", color="#e66101", region="flight lines", grid="10.6 m",
                   date="8 May 2025", kind="single day"),
}
FRAME_COLOR = "#444444"
NON_DRYLAND_GREY = "#e8e8e8"
NO_DATA_GREY = "#b0b0b0"

MM = 1 / 25.4
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 5.5,
    "axes.titlesize": 6,
    "axes.titleweight": "bold",
    "axes.titlepad": 2,
    "axes.labelsize": 5.5,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
    "legend.fontsize": 5,
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
PC = ccrs.PlateCarree()
ROBINSON = ccrs.Robinson()
MAP_LABEL_SIZE = 7        # lat/lon labels in the main figure
SI_LABEL_SIZE = 6         # lat/lon labels in the (denser) SI figure
MAP_FRAME = dict(color="black", lw=1.0)

# Page layout shared by Fig. 4 and its SI figure (inches)
FIG_W = 183 * MM       # Nature double-column width
HEADING = 0.27         # panel letter + title above a panel
SIDE = 0.30            # latitude labels beside the global map
LON_LABELS = 0.17      # longitude labels under a global map
TICKS = 0.2            # tick labels under a regional/local map
ROW2_H = 2.6           # height of panels b and c
SI_GRID = (("CI", "SCI"), ("SCAI", "BOCI"))   # order of the indices in each 2 x 2 SI panel


# ================================================================ data
def read_decimated(path, factor, bands=INDICES):
    """Average-resample the named bands by `factor`; nodata -> NaN. Returns (dict, transform, crs)."""
    with rasterio.open(path) as ds:
        shape = (ds.height // factor, ds.width // factor)
        out = {}
        for name in bands:
            arr = ds.read(ds.descriptions.index(name) + 1, out_shape=shape,
                          resampling=Resampling.average, masked=True)
            out[name] = arr.filled(np.nan).astype(np.float32)
        transform = ds.transform * ds.transform.scale(ds.width / shape[1], ds.height / shape[0])
        return out, transform, ds.crs


def dryland_class_on_grid(transform, shape):
    """UNEP aridity class (-1 = no data / ice / ocean) resampled to a lon/lat display grid."""
    dst = np.full(shape, -1, dtype=np.int8)
    with rasterio.open(ARIDITY_TIF) as ds:
        reproject(rasterio.band(ds, 1), dst, dst_transform=transform, dst_crs="EPSG:4326",
                  src_nodata=-1, dst_nodata=-1, resampling=Resampling.mode)
    return dst


def utm_to_lonlat(bands, transform, crs):
    """Reproject decimated UTM arrays onto a lon/lat grid of about the same pixel size."""
    h, w = next(iter(bands.values())).shape
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(h, w, transform))
    res = abs(transform.a) / 111_320 * 0.9
    dst_w, dst_h = int(np.ceil((east - west) / res)), int(np.ceil((north - south) / res))
    dst_transform = from_bounds(west, south, east, north, dst_w, dst_h)
    out = {}
    for name, arr in bands.items():
        dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        reproject(arr, dst, src_transform=transform, src_crs=crs, src_nodata=np.nan,
                  dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=Resampling.nearest)
        out[name] = dst
    return out, dst_transform


def load_display_arrays():
    """{platform: {index: array}} on lon/lat grids, their transforms, and the PACE dryland classes."""
    if os.path.exists(CACHE) and not REBUILD_CACHE:
        z = np.load(CACHE)
        data = {p: {n: z[f"{p}_{n}"] for n in INDICES} for p in ("PACE", "EMIT", "AVIRIS")}
        transforms = {p: rasterio.Affine(*z[f"{p}_transform"]) for p in data}
        return data, transforms, z["PACE_aridity"]

    print("reading GeoTIFFs at display resolution (slow; cached afterwards) ...")
    data, transforms = {}, {}
    data["PACE"], transforms["PACE"], _ = read_decimated(PACE_TIF, PACE_DECIMATE)
    data["EMIT"], transforms["EMIT"], _ = read_decimated(EMIT_TIF, EMIT_DECIMATE)
    utm, utm_transform, utm_crs = read_decimated(AVIRIS_TIF, AVIRIS_DECIMATE)
    data["AVIRIS"], transforms["AVIRIS"] = utm_to_lonlat(utm, utm_transform, utm_crs)
    aridity = dryland_class_on_grid(transforms["PACE"], data["PACE"]["CI"].shape)

    np.savez_compressed(CACHE, PACE_aridity=aridity,
                        **{f"{p}_{n}": a for p, d in data.items() for n, a in d.items()},
                        **{f"{p}_transform": np.array(t)[:6] for p, t in transforms.items()})
    return data, transforms, aridity


# ================================================================ composite colour
def square_position(bands):
    """Position (x, y) of each pixel in the unit-square key, from the relative balance of its indices.

    Each index is scaled to 0-1 with its fixed range and divided by the four-index sum (its share).
    x is the share held by the two right-hand corner indices and y the share of the two top ones, so a
    pixel where one index dominates sits at that index's corner and an even mix sits in the centre.
    """
    scaled = {n: np.clip((bands[n] - lo) / (hi - lo), 0, 1) for n, (lo, hi) in INDEX_RANGES.items()}
    total = sum(scaled.values())
    with np.errstate(invalid="ignore", divide="ignore"):
        share = {n: v / total for n, v in scaled.items()}
    x = sum(share[n] for n, ((cx, _), _) in CORNERS.items() if cx == 1)
    y = sum(share[n] for n, ((_, cy), _) in CORNERS.items() if cy == 1)
    return x, y


def key_colour(x, y):
    """Bilinear blend of the corner colours at (x, y)."""
    rgb = 0
    for (cx, cy), color in CORNERS.values():
        weight = (x if cx else 1 - x) * (y if cy else 1 - y)
        rgb = rgb + weight[..., None] * np.array(color)
    return np.clip(rgb, 0, 1)


def key_window(positions):
    """Zoomed key extent (x0, x1, y0, y1): KEY_ZOOM_PCT of pixel positions, equal weight per panel."""
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for x, y in positions:
        ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        pick = rng.choice(ok, min(300_000, ok.size), replace=False)
        xs.append(x.ravel()[pick])
        ys.append(y.ravel()[pick])
    x0, x1 = np.percentile(np.concatenate(xs), KEY_ZOOM_PCT)
    y0, y1 = np.percentile(np.concatenate(ys), KEY_ZOOM_PCT)
    return x0, x1, y0, y1


def composite_rgba(position, window):
    """RGBA image: key colour at each pixel's position, rescaled to the zoomed key window."""
    x, y = position
    x0, x1, y0, y1 = window
    xz = np.clip((x - x0) / (x1 - x0), 0, 1)
    yz = np.clip((y - y0) / (y1 - y0), 0, 1)
    valid = np.isfinite(xz) & np.isfinite(yz)
    return np.dstack([key_colour(np.nan_to_num(xz), np.nan_to_num(yz)), valid]).astype(np.float32)


def colormap_rgba(values, index, cmap="viridis"):
    """RGBA image of one index on its fixed range (for cartopy panels, which need RGBA arrays)."""
    lo, hi = INDEX_RANGES[index]
    rgba = plt.get_cmap(cmap)(np.clip((values - lo) / (hi - lo), 0, 1))
    rgba[..., 3] = np.isfinite(values)
    return rgba.astype(np.float32)


# ================================================================ map drawing
HALO = [pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()]
TEXT_HALO = [pe.withStroke(linewidth=1.2, foreground="white")]


def raster_extent(transform, shape):
    h, w = shape
    return [transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f]


def mean_lat(extent):
    return np.deg2rad((extent[2] + extent[3]) / 2)


def lonlat_aspect(extent):
    """Width / height of a lon/lat extent drawn with the local scale of its mean latitude."""
    return (extent[1] - extent[0]) * np.cos(mean_lat(extent)) / (extent[3] - extent[2])


def global_aspect():
    """Width / height of the full Robinson globe."""
    (x0, x1), (y0, y1) = ROBINSON.x_limits, ROBINSON.y_limits
    return (x1 - x0) / (y1 - y0)


def degree_label(value, pos, neg):
    value = round(value, 2)
    return f"{abs(value):g}°{neg if value < 0 else pos}"


def lonlat_ticks(ax, nbins=3, left=True, bottom=True, size=MAP_LABEL_SIZE):
    for axis, pos, neg in ((ax.xaxis, "E", "W"), (ax.yaxis, "N", "S")):
        axis.set_major_locator(MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10]))
        axis.set_major_formatter(FuncFormatter(lambda v, _, p=pos, n=neg: degree_label(v, p, n)))
    ax.tick_params(axis="both", labelsize=size, direction="out", length=2, width=0.5, pad=1.3)
    ax.tick_params(axis="y", labelrotation=90, labelleft=left)
    ax.tick_params(axis="x", labelbottom=bottom)
    for label in ax.get_yticklabels():
        label.set_verticalalignment("center")


def map_grid(ax):
    """Dashed grey graticule at the tick positions, drawn above the image."""
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(False)


def frame(ax, color, lw=0.9):
    spines = [ax.spines["geo"]] if "geo" in ax.spines else ax.spines.values()
    for spine in spines:
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def base_map(ax):
    """Natural Earth context layers: land, ocean, lakes, rivers, borders, coastlines, state lines."""
    states_provinces = cfeature.NaturalEarthFeature(category="cultural", name="admin_1_states_provinces_lines",
                                                    scale="50m", facecolor="none")
    ax.add_feature(cfeature.LAND, alpha=0.1, zorder=0)
    ax.add_feature(cfeature.BORDERS, linestyle="--", lw=0.4, alpha=0.5, zorder=4)
    ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=2)
    ax.add_feature(cfeature.OCEAN, alpha=0.1, zorder=2)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=4)
    ax.add_feature(cfeature.RIVERS, lw=0.2, zorder=4)
    ax.add_feature(states_provinces, lw=0.2, edgecolor="gray", zorder=4)


def global_gridlines(fig, ax, size=MAP_LABEL_SIZE, right=True):
    """Latitude labels from cartopy; flat longitude labels added under the map's bottom edge."""
    grd = ax.gridlines(draw_labels=True, xlocs=range(-180, 181, 90), ylocs=range(-60, 61, 30), color="gray",
                       linestyle="--", linewidth=0.5, zorder=7)
    grd.top_labels = False
    grd.right_labels = right
    grd.ylabel_style = {"size": size}
    grd.xformatter = FuncFormatter(lambda v, pos: "")   # cartopy's longitude labels follow the curved edge
    lat_bottom = ax.get_extent(PC)[2]
    below = mtransforms.offset_copy(ax.transData, fig=fig, y=-4, units="points")
    for lon, label in ((-180, "180°"), (-90, "90°W"), (0, "0°"), (90, "90°E"), (180, "180°")):
        x, y = ax.projection.transform_point(lon, lat_bottom, PC)
        ax.text(x, y, label, transform=below, ha="center", va="top", fontsize=size, zorder=8)


def draw_global(ax, rgba, transform, aridity, context=True):
    """Full-globe Robinson map: RGBA layer over drylands, drylands without a retrieval in grey.

    context=True draws the full base map (main figure); False only grey land and coastlines (SI panels).
    """
    if context:
        base_map(ax)
    else:
        ax.add_feature(cfeature.LAND, facecolor=NON_DRYLAND_GREY, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=0.2, edgecolor="#666666", zorder=4)
    no_data = np.zeros(aridity.shape + (4,), np.float32)
    no_data[np.isin(aridity, DRYLAND_CLASSES)] = to_rgba(NO_DATA_GREY)
    extent = raster_extent(transform, aridity.shape)
    for layer, z in ((no_data, 2.5), (rgba, 3)):
        ax.imshow(layer, extent=extent, transform=PC, origin="upper", interpolation="nearest",
                  regrid_shape=aridity.shape[0], zorder=z)
    ax.set_global()


def draw_lonlat(ax, image, extent, **imshow_kw):
    """Plain lon/lat axes drawn at the local scale of the mean latitude."""
    im = ax.imshow(image, extent=extent, origin="upper", interpolation="nearest", zorder=1, **imshow_kw)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(mean_lat(extent)))
    return im


def plot_lines(ax, geoms, **kw):
    for geom in geoms:
        for part in getattr(geom, "geoms", [geom]):
            ax.plot(*part.xy, **kw)


def context_lines(ax, extent):
    """US state lines and the Colorado Plateau outline inside a lon/lat extent."""
    from shapely.geometry import box
    bbox = box(extent[0], extent[2], extent[1], extent[3])
    states = shapereader.natural_earth("50m", "cultural", "admin_1_states_provinces_lines")
    plot_lines(ax, [g for g in shapereader.Reader(states).geometries() if g.intersects(bbox)],
               color="#777777", lw=0.3, ls=(0, (1, 1.2)), zorder=2)
    plot_lines(ax, [g.exterior for g in gpd.read_file(CO_PLATEAU).geometry.iloc[0].geoms],
               color="black", lw=0.45, zorder=3)


def coverage_outline(bands, transform, block=4):
    """Lon/lat grid and filled valid-data mask outlining where a scene has data (gaps closed)."""
    valid = np.all([np.isfinite(a) for a in bands.values()], axis=0)
    hb, wb = valid.shape[0] // block, valid.shape[1] // block
    coarse = valid[:hb * block, :wb * block].reshape(hb, block, wb, block).any(axis=(1, 3))
    coarse = ndimage.binary_closing(np.pad(coarse, 3), iterations=2)[3:-3, 3:-3]
    coarse = ndimage.binary_fill_holes(coarse)
    lon = transform.c + (np.arange(wb) + 0.5) * block * transform.a
    lat = transform.f + (np.arange(hb) + 0.5) * block * transform.e
    return lon, lat, coarse


def scale_bar(ax, km, loc=(0.06, 0.04)):
    """Horizontal scale bar in lon/lat axes, sized at the axes' mean latitude."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    length = km / (111.32 * np.cos(np.deg2rad((y0 + y1) / 2)))
    x, y = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], color="black", lw=1.0, solid_capstyle="butt", path_effects=HALO, zorder=6)
    ax.text(x + length / 2, y + 0.015 * (y1 - y0), f"{km} km", ha="center", va="bottom", fontsize=6,
            path_effects=TEXT_HALO, zorder=6)


def nest_label(ax, x, y, label, **kw):
    """Letter of the panel that shows the outlined area, at the outline's top-left corner."""
    ax.text(x, y, label, ha="right", va="bottom", fontsize=7, fontweight="bold", path_effects=TEXT_HALO,
            zorder=7, **kw)


def panel_heading(ax, letter, platform):
    """'a  ■ Platform · region' above the axes' top-left corner; the square carries the platform colour."""
    info = PLATFORMS[platform]
    common = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points", va="bottom", ha="left")
    ax.annotate(letter, xytext=(0, 3), fontsize=PANEL_LETTER_SIZE, fontweight="bold", **common)
    ax.annotate("■", xytext=(9, 3.2), fontsize=PANEL_TITLE_SIZE, color=info["color"], **common)
    ax.annotate(f"{info['label']} · {info['region']}", xytext=(18, 3.2), fontsize=PANEL_TITLE_SIZE, fontweight="bold",
                **common)


def draw_key(ax, fontsize=6.5):
    """Rectangle colour key with the index named at each corner."""
    X, Y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    ax.imshow(key_colour(X, Y), origin="lower", extent=[0, 1, 0, 1], interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    for index, ((cx, cy), _) in CORNERS.items():
        ax.text(cx, cy + (0.05 if cy else -0.05), INDEX_LABELS[index], ha="center",
                va="bottom" if cy else "top", fontsize=fontsize, fontweight="bold")


def axes_in(fig, left, bottom, width, height, **kw):
    """Axes placed in inches from the figure's lower-left corner."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kw)


def save(fig, name):
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(out, dpi=1000, facecolor="white")
        print("wrote", out)
    plt.close(fig)


# ================================================================ figures
def fig4_size():
    """Canvas of Fig. 4 (also used for its SI figure): full-width globe on top, row b-c below."""
    a_h = (FIG_W - 2 * SIDE) / global_aspect()
    return FIG_W, HEADING + a_h + LON_LABELS + HEADING + ROW2_H + TICKS + 0.03


def draw_legend_column(fig, left, bottom, width, height):
    """Colour key with the map symbols centred under it, then the platform table (inches)."""
    W, H = fig.get_size_inches()
    top, centre = bottom + height, left + width / 2

    key = 0.82
    key_top = top - 0.13                       # room for the top corner labels
    draw_key(axes_in(fig, centre - key / 2, key_top - key, key, key))

    land = to_rgba(cfeature.COLORS["land"], 0.1)
    fig.legend(handles=[Patch(facecolor=land, edgecolor="#c8c8c8", lw=0.3, label="Land outside drylands (a)"),
                        Patch(facecolor=NO_DATA_GREY, label="No valid retrieval (a–c)"),
                        Line2D([], [], color="black", lw=0.45, label="Colorado Plateau (b)")],
               loc="upper center", bbox_to_anchor=(centre / W, (key_top - key - 0.17) / H), frameon=False,
               fontsize=6, handlelength=1.8, handleheight=0.9, borderaxespad=0, labelspacing=0.4)

    columns = ((0.15, "Platform", "label"), (0.68, "Grid", "grid"), (1.30, "Acquisition", None))
    x0 = centre - 1.97 / 2                     # table is ~1.97 in wide at 6 pt
    table_top, row = bottom + 0.88, 0.24
    for dx, header, _ in columns:
        fig.text((x0 + dx) / W, table_top / H, header, fontsize=6, fontweight="bold", va="top")
    for r, info in enumerate(PLATFORMS.values()):
        y = (table_top - 0.14 - r * row) / H
        fig.text(x0 / W, y, "■", fontsize=7, color=info["color"], va="top")
        for dx, _, field in columns:
            value = info[field] if field else f"{info['date']}\n{info['kind']}"
            fig.text((x0 + dx) / W, y, value, fontsize=6, va="top", linespacing=1.2)


def plot_main(data, transforms, aridity, rgba):
    """a across the top (full globe); b and c side by side below it; legend column right of c."""
    W, H = fig4_size()
    fig = plt.figure(figsize=(W, H))

    # a: PACE, full globe; drylands coloured; box b = Colorado Plateau
    a_w = W - 2 * SIDE
    a_h = a_w / global_aspect()
    ax_a = axes_in(fig, SIDE, H - HEADING - a_h, a_w, a_h, projection=ROBINSON)
    draw_global(ax_a, rgba["PACE"], transforms["PACE"], aridity)
    global_gridlines(fig, ax_a)
    ax_a.spines["geo"].set_linewidth(MAP_FRAME["lw"])
    west, south, east, north = gpd.read_file(CO_PLATEAU).total_bounds
    pad = 1.5
    ax_a.plot([west - pad, east + pad, east + pad, west - pad, west - pad],
              [south - pad, south - pad, north + pad, north + pad, south - pad],
              color="black", lw=0.7, path_effects=HALO, zorder=6, transform=PC)
    nest_label(ax_a, west - pad, north + pad, "b", transform=PC)

    # b and c share the row height; their widths follow from each map's aspect
    b_ext = raster_extent(transforms["EMIT"], rgba["EMIT"].shape[:2])
    c_ext = raster_extent(transforms["AVIRIS"], rgba["AVIRIS"].shape[:2])
    row2_y = TICKS + 0.03
    b_x, b_w = 0.62, ROW2_H * lonlat_aspect(b_ext)
    c_x, c_w = b_x + b_w + 0.42, ROW2_H * lonlat_aspect(c_ext)
    ax_b = axes_in(fig, b_x, row2_y, b_w, ROW2_H)
    ax_c = axes_in(fig, c_x, row2_y, c_w, ROW2_H)

    # b: EMIT over the Colorado Plateau (grey inside it = no valid retrieval); outline c = AVIRIS-5 coverage
    for poly in gpd.read_file(CO_PLATEAU).geometry.iloc[0].geoms:
        ax_b.fill(*poly.exterior.xy, color=NO_DATA_GREY, lw=0, zorder=0)
    draw_lonlat(ax_b, rgba["EMIT"], b_ext)
    context_lines(ax_b, b_ext)
    lon, lat, cover = coverage_outline(data["AVIRIS"], transforms["AVIRIS"])
    outline = ax_b.contour(lon, lat, cover.astype(float), levels=[0.5], colors="black", linewidths=0.7, zorder=6)
    outline.set_path_effects(HALO)
    top_row = np.flatnonzero(cover.any(axis=1))[0]                   # northernmost covered row
    nest_label(ax_b, lon[cover[top_row]].min(), lat[top_row], "c")
    ax_b.set_xlim(b_ext[0], b_ext[1])
    ax_b.set_ylim(b_ext[2], b_ext[3])
    lonlat_ticks(ax_b, nbins=4)
    map_grid(ax_b)
    scale_bar(ax_b, 100)

    # c: AVIRIS-5, all flight lines (grey inside the coverage = no valid retrieval)
    ax_c.contourf(lon, lat, cover.astype(float), levels=[0.5, 1.5], colors=[NO_DATA_GREY], zorder=0)
    draw_lonlat(ax_c, rgba["AVIRIS"], c_ext)
    lonlat_ticks(ax_c, nbins=2)
    map_grid(ax_c)
    scale_bar(ax_c, 20)

    for ax, letter, platform in ((ax_a, "a", "PACE"), (ax_b, "b", "EMIT"), (ax_c, "c", "AVIRIS")):
        if ax is not ax_a:
            frame(ax, **MAP_FRAME)
        panel_heading(ax, letter, platform)

    legend_x = c_x + c_w + 0.2
    draw_legend_column(fig, legend_x, row2_y, W - legend_x - 0.05, ROW2_H)
    save(fig, "Fig4")


def index_tag(ax, index):
    """Short index name in the map's top-left corner."""
    ax.text(0.02, 0.98, INDEX_LABELS[index], transform=ax.transAxes, ha="left", va="top", fontsize=7,
            fontweight="bold", path_effects=TEXT_HALO, zorder=9)


def plot_si(data, transforms, aridity):
    """Each index on its own, on the Fig. 4 canvas: a (PACE) on top, b (EMIT) and c (AVIRIS-5) below.

    Every panel is a 2 x 2 block in SI_GRID order; one fixed colour range per index, colour bars on the right.
    """
    W, H = fig4_size()
    fig = plt.figure(figsize=(W, H))
    lat_room, gap = 0.28, 0.08                 # rotated latitude labels left of every map; gap between maps

    # a: four globes, centred
    a_h = 1.42
    a_w = a_h * global_aspect()
    a_x = (W - 2 * (lat_room + a_w) - gap) / 2 + lat_room
    y = H - HEADING
    first = None
    for row in SI_GRID:
        y -= a_h
        for c, index in enumerate(row):
            ax = axes_in(fig, a_x + c * (a_w + lat_room + gap), y, a_w, a_h, projection=ROBINSON)
            draw_global(ax, colormap_rgba(data["PACE"][index], index), transforms["PACE"], aridity, context=False)
            global_gridlines(fig, ax, size=SI_LABEL_SIZE, right=False)
            ax.spines["geo"].set_linewidth(0.8)
            index_tag(ax, index)
            first = first or ax
        y -= LON_LABELS + 0.03
    panel_heading(first, "a", "PACE")

    # b, c: 2 x 2 blocks side by side, equal map height
    bc_top = y - HEADING
    bc_h = (bc_top - 0.03 - 2 * TICKS - 0.04) / 2
    lon, lat, cover = coverage_outline(data["AVIRIS"], transforms["AVIRIS"])
    x = lat_room + 0.05
    for key, letter, nbins in (("EMIT", "b", 3), ("AVIRIS", "c", 2)):
        ext = raster_extent(transforms[key], data[key]["CI"].shape)
        w = bc_h * lonlat_aspect(ext)
        first = None
        for r, row in enumerate(SI_GRID):
            y = bc_top - (r + 1) * bc_h - r * (TICKS + 0.04)
            for c, index in enumerate(row):
                ax = axes_in(fig, x + c * (w + lat_room + gap), y, w, bc_h)
                if key == "EMIT":
                    for poly in gpd.read_file(CO_PLATEAU).geometry.iloc[0].geoms:
                        ax.fill(*poly.exterior.xy, color=NO_DATA_GREY, lw=0, zorder=0)
                else:
                    ax.contourf(lon, lat, cover.astype(float), levels=[0.5, 1.5], colors=[NO_DATA_GREY], zorder=0)
                lo, hi = INDEX_RANGES[index]
                draw_lonlat(ax, np.ma.masked_invalid(data[key][index]), ext, cmap="viridis", vmin=lo, vmax=hi)
                if key == "EMIT":
                    context_lines(ax, ext)
                    ax.set_xlim(ext[0], ext[1])
                    ax.set_ylim(ext[2], ext[3])
                lonlat_ticks(ax, nbins=nbins, size=SI_LABEL_SIZE)
                map_grid(ax)
                frame(ax, MAP_FRAME["color"], lw=0.8)
                index_tag(ax, index)
                first = first or ax
        panel_heading(first, letter, key)
        x += 2 * w + lat_room + gap + 0.08 + lat_room

    # one vertical colour bar per index, side by side right of c; full name along each bar
    cb_left = x - lat_room + 0.08
    unit = (W - 0.05 - cb_left) / 4
    cb_bottom, cb_top = TICKS + 0.1, bc_top - 0.05
    for k, index in enumerate(i for row in SI_GRID for i in row):
        cax = axes_in(fig, cb_left + k * unit + 0.15, cb_bottom, 0.075, cb_top - cb_bottom)
        lo, hi = INDEX_RANGES[index]
        cb = fig.colorbar(ScalarMappable(Normalize(lo, hi), "viridis"), cax=cax, orientation="vertical",
                          extend="both", extendfrac=0.03)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, width=0.4, length=1.5, pad=1)
        cax.yaxis.set_label_position("left")
        cb.set_label(rf"$\mathbf{{{INDEX_LABELS[index]}}}$ · {INDEX_NAMES[index].lower()}", fontsize=7, labelpad=2)
    save(fig, "Fig4_SI_single_indices")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(NATURE_RC)
    data, transforms, aridity = load_display_arrays()
    dry = np.isin(aridity, DRYLAND_CLASSES)
    data["PACE"] = {n: np.where(dry, a, np.nan) for n, a in data["PACE"].items()}   # drylands only

    positions = {p: square_position(d) for p, d in data.items()}
    window = key_window(positions.values())
    print("key window (share of right / top corner indices): x {:.3f}-{:.3f}, y {:.3f}-{:.3f}".format(*window))
    rgba = {p: composite_rgba(positions[p], window) for p in data}
    plot_main(data, transforms, aridity, rgba)
    plot_si(data, transforms, aridity)


if __name__ == "__main__":
    main()
