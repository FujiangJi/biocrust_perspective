"""Reclassify TERRACLIMATE Aridity Index into 5 UNEP classes (0-4), save as int8 GeoTIFF,
export a Western-US clip, AND render a publication-quality global aridity/dryland map.

Class scheme (UNEP):
    0 = Humid        (AI >= 0.65)
    1 = Sub-humid    (0.50 <= AI < 0.65)      \
    2 = Semi-arid    (0.20 <= AI < 0.50)       > drylands (AI < 0.65)
    3 = Arid         (0.05 <= AI < 0.20)       |
    4 = Hyper-arid   (AI < 0.05)              /
   -1 = NoData       (sentinel)

Outputs:
    TERRACLIMATE_ARIDITY_GLOBAL_GEO_4km_classes.tif   (full global, int8)
    TERRACLIMATE_ARIDITY_WESTERN_US_4km_classes.tif   (clipped to Western US, int8)
    ../figures/global_aridity_map.png / .pdf          (Robinson global map)
    ../figures/global_aridity_map_drylands.png / .pdf (dryland-emphasis version)

Run with the geo environment:
    /Users/jifujiang/opt/anaconda3/envs/Fujiang/bin/python ariditymap_processing.py
"""
import os
import numpy as np
import rasterio as rs
from rasterio.mask import mask as rio_mask
from rasterio.features import rasterize
from shapely.geometry import box, mapping
from shapely.ops import unary_union

import matplotlib
matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
DATA_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/0_src_code/1_data"
SRC_FILE = "TERRACLIMATE_ARIDITY_GLOBAL_GEO_4km.tif"
DST_GLOBAL = "TERRACLIMATE_ARIDITY_GLOBAL_GEO_4km_classes.tif"
DST_NOICE = "TERRACLIMATE_ARIDITY_GLOBAL_GEO_4km_classes_noice.tif"  # polar ice masked out
DST_WESTERN = "TERRACLIMATE_ARIDITY_WESTERN_US_4km_classes.tif"
NODATA_OUT = np.int8(-1)

# Where the perspective-paper figures go (sibling of 0_src_code)
FIG_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/figures"

# Re-run the (slow, 105 MB read) reclassification only when needed.
REBUILD_CLASSES = False
# Decimate factor for the display raster (nearest); plenty of detail for a global figure.
DISPLAY_DECIMATE = 4

# Mask permanent ice (Antarctica, Greenland, glaciers) so polar cold deserts do not
# inflate the dryland fraction — these are not biocrust habitat. Set False for the
# Aridity-Index-faithful full-global version.
MASK_POLAR_ICE = True
LAT_S_CUT = -60.0   # drop everything south of this (Antarctica)
MAP_LAT_S, MAP_LAT_N = -58.0, 84.0  # display crop (Robinson)

os.chdir(DATA_DIR)

# UNEP class metadata (single source of truth for labels / colors / tags)
CLASSES = [
    # value, short label, legend label,                 color (wet -> dry)
    (0, "Humid",      "Humid",     "#1a9850"),
    (1, "Sub-humid",  "Sub-humid", "#a6d96a"),
    (2, "Semi-arid",  "Semi-arid",     "#fee08b"),
    (3, "Arid",       "Arid",          "#fdae61"),
    (4, "Hyper-arid", "Hyper-arid",     "#d73027"),
]
LABELS = {-1: "NoData", **{v: s for v, s, _, _ in CLASSES}}

CLASS_TAGS = dict(
    AI_CLASS_0="Humid (AI >= 0.65)",
    AI_CLASS_1="Sub-humid (0.50 <= AI < 0.65)",
    AI_CLASS_2="Semi-arid (0.20 <= AI < 0.50)",
    AI_CLASS_3="Arid (0.05 <= AI < 0.20)",
    AI_CLASS_4="Hyper-arid (AI < 0.05)",
    NODATA="-1",
    SOURCE=SRC_FILE,
)


# =========================================================================
# 1. Reclassify global AI -> 5 UNEP classes  (guarded: skip if up to date)
# =========================================================================
def build_classes():
    with rs.open(SRC_FILE) as src:
        ai = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        src_nodata = src.nodata

    nodata_mask = np.isnan(ai) | (ai < 0)
    if src_nodata is not None:
        nodata_mask |= (ai == np.float32(src_nodata))

    conditions = [
        ai >= 0.65,                   # 0 Humid
        (ai >= 0.50) & (ai < 0.65),   # 1 Sub-humid
        (ai >= 0.20) & (ai < 0.50),   # 2 Semi-arid
        (ai >= 0.05) & (ai < 0.20),   # 3 Arid
        ai < 0.05,                    # 4 Hyper-arid
    ]
    classes = np.select(conditions, [0, 1, 2, 3, 4], default=NODATA_OUT).astype(np.int8)
    classes[nodata_mask] = NODATA_OUT

    profile.update(
        dtype="int8", nodata=int(NODATA_OUT), compress="deflate", predictor=2,
        tiled=True, blockxsize=512, blockysize=512,
    )
    with rs.open(DST_GLOBAL, "w", **profile) as dst:
        dst.write(classes, 1)
        dst.update_tags(**CLASS_TAGS)
    print(f"Reclassified -> {DST_GLOBAL}")


if REBUILD_CLASSES or not os.path.exists(DST_GLOBAL):
    build_classes()
else:
    print(f"Reusing existing {DST_GLOBAL} (set REBUILD_CLASSES=True to rebuild).")


# =========================================================================
# 1b. Mask permanent ice (Antarctica / Greenland / glaciers) -> DST_NOICE
# =========================================================================
def load_ice_geoms():
    """Natural Earth permanent-ice polygons; empty list + Greenland bbox fallback."""
    try:
        import cartopy.io.shapereader as shpreader
        geoms = []
        for name in ("glaciated_areas", "antarctic_ice_shelves_polys"):
            try:
                shp = shpreader.natural_earth(resolution="50m", category="physical", name=name)
                geoms += [rec.geometry for rec in shpreader.Reader(shp).records()]
            except Exception:
                pass
        if geoms:
            print(f"Loaded {len(geoms)} permanent-ice polygons from Natural Earth.")
            return geoms
    except Exception as e:
        print(f"Ice-polygon load failed ({e}); using Greenland bbox fallback.")
    return [box(-73.0, 59.0, -11.0, 84.0)]  # Greenland ice sheet (Antarctica handled by LAT_S_CUT)


def build_noice():
    with rs.open(DST_GLOBAL) as src:
        classes = src.read(1).copy()
        profile = src.profile.copy()
        transform = src.transform
        H, W = src.height, src.width
        top, bottom = src.bounds.top, src.bounds.bottom

    mask = np.zeros((H, W), dtype=bool)
    geoms = load_ice_geoms()
    if geoms:
        mask |= rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=transform,
                          fill=0, dtype="uint8").astype(bool)
    lats = np.linspace(top, bottom, H)
    mask |= (lats < LAT_S_CUT)[:, None]  # Antarctica

    classes[mask] = int(NODATA_OUT)
    profile.update(dtype="int8", nodata=int(NODATA_OUT), compress="deflate", predictor=2,
                   tiled=True, blockxsize=512, blockysize=512)
    with rs.open(DST_NOICE, "w", **profile) as dst:
        dst.write(classes, 1)
        dst.update_tags(**CLASS_TAGS, NOTE="Permanent ice (Antarctica/Greenland/glaciers) masked to NoData")
    print(f"Ice-masked -> {DST_NOICE}")


if MASK_POLAR_ICE and (REBUILD_CLASSES or not os.path.exists(DST_NOICE)):
    build_noice()

# The raster that the stats + figure use.
MAP_SRC = DST_NOICE if MASK_POLAR_ICE else DST_GLOBAL


# =========================================================================
# 2. Build Western-US polygon (Natural Earth states; fall back to bbox)
# =========================================================================
def western_us_geom():
    western_states = {
        "California", "Oregon", "Washington", "Idaho", "Nevada", "Utah", "Arizona",
        "Montana", "Wyoming", "Colorado", "New Mexico",
        "North Dakota", "South Dakota", "Nebraska", "Kansas", "Oklahoma", "Texas",
    }
    bbox = box(-125.0, 25.0, -94.0, 50.0)
    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                      name="admin_1_states_provinces")
        geoms = [rec.geometry for rec in shpreader.Reader(shp).records()
                 if rec.attributes.get("admin") == "United States of America"
                 and rec.attributes.get("name") in western_states]
        if geoms:
            g = unary_union(geoms).intersection(bbox)
            print(f"Built Western-US polygon from {len(geoms)} state features.")
            return g if not g.is_empty else bbox
    except Exception as e:
        print(f"Natural Earth load failed ({e}); using bounding box.")
    return bbox


# =========================================================================
# 3. Clip the classified raster to Western US and save  (guarded)
# =========================================================================
if REBUILD_CLASSES or not os.path.exists(DST_WESTERN):
    geom = western_us_geom()
    with rs.open(DST_GLOBAL) as src:
        out_image, out_transform = rio_mask(
            src, [mapping(geom)], crop=True, nodata=int(NODATA_OUT), filled=True,
        )
        out_profile = src.profile.copy()
    out_profile.update(
        height=out_image.shape[1], width=out_image.shape[2], transform=out_transform,
        dtype="int8", nodata=int(NODATA_OUT), compress="deflate", predictor=2,
        tiled=True, blockxsize=512, blockysize=512,
    )
    with rs.open(DST_WESTERN, "w", **out_profile) as dst:
        dst.write(out_image[0].astype(np.int8), 1)
        dst.update_tags(**CLASS_TAGS, EXTENT="Western US", BBOX="-125 25 -94 50")
    print(f"Clipped -> {DST_WESTERN}")
else:
    print(f"Reusing existing {DST_WESTERN}.")


# =========================================================================
# 4. Sanity check — area-weighted (cos-lat) class fractions
# =========================================================================
def summarize(path):
    with rs.open(path) as src:
        arr = src.read(1)
        h = src.height
        top, bottom = src.bounds.top, src.bounds.bottom
    lats = np.linspace(top, bottom, h)                       # per-row latitude
    w = np.cos(np.deg2rad(lats))[:, None] * np.ones((1, arr.shape[1]))
    valid = arr != int(NODATA_OUT)
    land_w = w[valid].sum()
    print(f"\n{path}  ({arr.shape[1]}x{arr.shape[0]})  [area-weighted over land]")
    for v in [-1, 0, 1, 2, 3, 4]:
        m = arr == v
        if not m.any():
            continue
        note = "" if v == -1 else f"({100.0 * w[m].sum() / land_w:5.2f}% of land)"
        print(f"  {v:>3d}  {LABELS[v]:<10s}  {int(m.sum()):>12,d} px  {note}")
    return arr, w, valid, land_w


arr_g, w_g, valid_g, land_w_g = summarize(MAP_SRC)
dry = np.isin(arr_g, [1, 2, 3, 4])
DRY_PCT = 100.0 * w_g[dry].sum() / land_w_g
print(f"\n  Drylands (AI < 0.65, classes 1-4): {DRY_PCT:.1f}% of land area"
      f"{'  [polar ice masked]' if MASK_POLAR_ICE else ''}")
summarize(DST_WESTERN)


# =========================================================================
# 5. Render the global aridity / dryland map  (the figure deliverable)
# =========================================================================
def render_global_map(highlight_drylands=False):
    """Robinson global map of UNEP aridity classes.
    highlight_drylands=True fades humid areas to grey so drylands stand out."""
    os.makedirs(FIG_DIR, exist_ok=True)

    with rs.open(MAP_SRC) as src:
        h = src.height // DISPLAY_DECIMATE
        wdt = src.width // DISPLAY_DECIMATE
        data = src.read(1, out_shape=(h, wdt), resampling=rs.enums.Resampling.nearest)
        b = src.bounds
    extent = [b.left, b.right, b.bottom, b.top]

    colors = [c for _, _, _, c in CLASSES]
    if highlight_drylands:
        colors = ["#e8e8e8", "#a6d96a", "#fee08b", "#fdae61", "#d73027"]  # humid -> grey

    cmap = ListedColormap(colors)
    cmap.set_bad(alpha=0.0)  # NoData transparent
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    masked = np.ma.masked_equal(data, int(NODATA_OUT))

    BG = "white"  # ocean == figure background
    fig = plt.figure(figsize=(10, 5), dpi=300, facecolor=BG)
    ax = plt.axes(projection=ccrs.Robinson())
    if MASK_POLAR_ICE:
        ax.set_extent([-180, 180, MAP_LAT_S, MAP_LAT_N], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    # No frame (spine) and ocean blended into the page background.
    ax.set_facecolor(BG)                          # GeoAxes.patch == ocean/background fill
    try:                                          # preferred (avoids outline_patch deprecation)
        ax.spines["geo"].set_visible(False)
    except Exception:
        if hasattr(ax, "outline_patch"):          # very old cartopy fallback
            ax.outline_patch.set_visible(False)

    ax.imshow(masked, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
              cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.4,
                   edgecolor="#555555", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.2,
                   edgecolor="#999999", zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#cccccc",
                 alpha=0.6, linestyle="--")

    handles = [Patch(facecolor=colors[i], edgecolor="#444444", linewidth=0.4, label=lbl)
               for i, (_, _, lbl, _) in enumerate(CLASSES)]
    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.63, 0.0),
                    frameon=False, ncol=5, columnspacing=0.8,
                    fontsize=9, borderpad=0.8, handlelength=1.4)
    # leg.get_frame().set_edgecolor("#cccccc")

    # title = "Global distribution of drylands" if highlight_drylands else \
    #         "Global aridity distribution (UNEP Aridity Index)"
    # ax.set_title(title, fontsize=15, pad=12, weight="bold")
    # icenote = ", permanent ice excluded" if MASK_POLAR_ICE else ""
    # ax.text(0.5, -0.06,
    #         f"Drylands (AI < 0.65) cover {DRY_PCT:.0f}% of Earth's land surface  ·  "
    #         f"Source: TerraClimate Aridity Index (~4 km{icenote})",
    #         transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#555555")

    suffix = "_drylands" if highlight_drylands else ""
    png = os.path.join(FIG_DIR, f"global_aridity_map{suffix}.png")
    pdf = os.path.join(FIG_DIR, f"global_aridity_map{suffix}.pdf")
    fig.savefig(png, bbox_inches="tight", dpi=1000, facecolor=BG)
    fig.savefig(pdf, bbox_inches="tight", dpi=1000, facecolor=BG)
    plt.close(fig)
    print(f"Wrote figure: {png}\n              {pdf}")


render_global_map(highlight_drylands=False)   # full 5-class UNEP map
render_global_map(highlight_drylands=True)    # dryland-emphasis version

print(f"\nDone.\n  Rasters in: {DATA_DIR}\n  Figures in: {FIG_DIR}")
