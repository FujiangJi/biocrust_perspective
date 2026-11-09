"""Figure 3: the global biocrust spectral library - the same spectral regions matter at every site, and the
biocrust indices are built from them.

a  (a.1-a.8) one panel per site: class-mean reflectance of the biocrust types and bare substrate measured there,
   dry (solid) and wet (dashed), ± 1 SD across replicate spectra where the workbook provides them (Negev 20,
   Spain 4 plot means per class, Wajir 13 biocrust / 16 soil). The strip above each panel:
   black ticks = the site's own dominant wavelengths from its PLS-R model (Karnieli et al., in review, Table 4);
   amber segments (identical in every panel) = the range of those wavelengths across the eight sites, i.e. the
   NUV, blue, red, NIR and SWIR2 regions that are preserved globally.
b  (b.1-b.4) biocrust indices CI, SCI, BOMI and SCAI (Karnieli et al., Eqs 1, 3, 4, 5) computed from each site's
   class-mean spectra, dry and wet pooled (one value per site and condition), by biocrust type: box in the type's
   colour across all values, symbols identify the site (colour + shape); n = number of spectra. The formula under
   each title names the bands, i.e. the shaded regions of a (green and SWIR1 are reference bands).

Data: 1_data/3_Fig3_data/0_biocrust_global_spectra (workbooks compiled by Arnon Karnieli). Negev and Kenya
list biocrust and substrate samples without a type; both sites are cyanobacteria-dominated (Karnieli et al.,
Table 1), so their biocrusts are plotted as cyanobacteria. The Negev workbook's two 20-column blocks and the
Kenya B*/K* sample groups were assigned to biocrust/substrate from brightness and the 680-nm chlorophyll
absorption (biocrusts darker, stronger absorption). Colorado Plateau has dry spectra only.

Output: figures/Fig3.{png,pdf}; 183 mm wide, Arial, 1000 dpi, editable text in the PDF.
Run with the Fujiang env:
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_fig3.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

import plot_fig2 as f2

SPEC_DIR = os.path.join(os.path.dirname(f2.DATA_DIR), "3_Fig3_data", "0_biocrust_global_spectra")
FIG_DIR = f2.FIG_DIR
OUT_NAME = "Fig3"

MM = 1 / 25.4
FIG_W = 183 * MM
NATURE_RC = f2.NATURE_RC
PANEL_LETTER_SIZE = f2.PANEL_LETTER_SIZE
PANEL_TITLE_SIZE = f2.PANEL_TITLE_SIZE
BAD_BANDS = f2.BAD_BANDS
WL = np.arange(350, 2501)          # common wavelength grid (nm)
XLIM = (350, 2500)

CLASSES = ["Cyanobacteria", "Lichen", "Moss", "Bare soil"]
CLASS_COLORS = {"Cyanobacteria": "#ff7f00", "Lichen": "#00bfc4", "Moss": "#984ea3", "Bare soil": "#666666"}
COND_STYLES = {"Dry": "-", "Wet": "--"}

# ---------------------------------------------------------------- sites
# Order = Karnieli et al. Table 1 = panels a.1-a.8. lonlat from Table 1 (used by plot_fig7 for the map stars).
# blocks: (sheet, header row, wavelength column, scale to 0-1 reflectance, {(class, condition): columns}[, reps]);
# columns are names or integer positions and are averaged -> one class mean per site. reps=True marks columns
# that are replicate spectra (SD band drawn); otherwise they are class/subclass means (no band).
KENYA_B = lambda c: [f"B{i}{c}" for i in range(1, 14)]
KENYA_K = lambda c: [f"K{p}{t}{k}_{c}" for p in (3, 5, 8, 10) for t in "CM" for k in (1, 2)]
SITES = {
    "Negev": dict(full="Negev Desert, Israel", lonlat=(34.360, 31.058),
                  file="Negev Desert final.xlsx", blocks=[
        ("REF_DRY", 0, 0, 1, {("Cyanobacteria", "Dry"): list(range(1, 21)), ("Bare soil", "Dry"): list(range(22, 42))}, True),
        ("REF_WET", 0, 0, 1, {("Cyanobacteria", "Wet"): list(range(1, 21)), ("Bare soil", "Wet"): list(range(22, 42))}, True),
    ]),
    "Namib": dict(full="Namib Desert, Namibia", lonlat=(15.0, -23.0),
                  file="Namibia_Ref_all.xlsx", blocks=[
        ("Figures2", 9, "Wavelength", 1, {("Lichen", "Dry"): ["Dry Lichens"], ("Lichen", "Wet"): ["Wet Lichens"],
                                          ("Bare soil", "Dry"): ["Dry Sands"], ("Bare soil", "Wet"): ["Wet Sands"]}),
    ]),
    "Colorado Plateau": dict(full="Colorado Plateau, USA", lonlat=(-109.150, 38.033),
                             file="Colordo Plateau.xlsx", blocks=[
        ("VIP_REGULAR_TRAN_1cyan_NWA", 8, 27, 1, {("Cyanobacteria", "Dry"): [29, 30], ("Lichen", "Dry"): [32],
                                                  ("Moss", "Dry"): [31], ("Bare soil", "Dry"): [28]}),
    ]),
    # Spain: the four plot means (P1/P2 x a/b) per class are the replicates; the workbook's own class-mean
    # columns ("Dry cyanobacteria", ...) sit right after each group and are checked against them (verify_spain)
    "Las Amoladeras": dict(full="Las Amoladeras, Spain", lonlat=(-2.252, 36.834),
                           file="Spain AMOL VIP_scores.xlsx", blocks=[
        ("VIP_scores", 8, "Label", 0.01, {
            ("Cyanobacteria", "Dry"): [18, 19, 20, 21], ("Cyanobacteria", "Wet"): [45, 46, 47, 48],
            ("Lichen", "Dry"): [24, 25, 26, 27], ("Lichen", "Wet"): [51, 52, 53, 54],
            ("Moss", "Dry"): [30, 31, 32, 33], ("Moss", "Wet"): [57, 58, 59, 60],
            ("Bare soil", "Dry"): [36, 37, 38, 39], ("Bare soil", "Wet"): [63, 64, 65, 66]}, True),
    ]),
    "El Cautivo": dict(full="El Cautivo, Spain", lonlat=(-2.442, 37.010),
                       file="Spain_CAUT_VIP.xlsx", blocks=[
        ("VIP_CAUT", 8, "Label", 0.01, {
            ("Cyanobacteria", "Dry"): [16, 17, 18, 19], ("Cyanobacteria", "Wet"): [37, 38, 39, 40],
            ("Lichen", "Dry"): [22, 23, 24, 25], ("Lichen", "Wet"): [43, 44, 45, 46],
            ("Bare soil", "Dry"): [28, 29, 30, 31], ("Bare soil", "Wet"): [49, 50, 51, 52]}, True),
    ]),
    "Soebatsfontein": dict(full="Soebatsfontein, South Africa", lonlat=(17.9, -29.6),
                           file="Soebatsfontein District SA.xlsx", blocks=[
        ("SA_VIP_8classes", 6, "sample", 0.01, {
            ("Cyanobacteria", "Dry"): ["Dry Cyano", "Dry Dark Cyano", "Dry Cyano+Dark"],
            ("Cyanobacteria", "Wet"): ["Wet Cyano", "Wet Dark Cyano", "Wet Cyano+Dark"],
            ("Lichen", "Dry"): ["Dry Lichens", "Dry Cyanolichens", "Dry Chlorolichens"],
            ("Lichen", "Wet"): ["Wet Lichens", "Wet Cyanolichens", "Wet Chlorolichens"],
            ("Moss", "Dry"): ["Dry Mosses"], ("Moss", "Wet"): ["Wet Mosses"],
            ("Bare soil", "Dry"): ["Dry Soil"], ("Bare soil", "Wet"): ["Wet Soil"]}),
    ]),
    "Gurbantünggüt": dict(full="Gurbantünggüt Desert, China", lonlat=(87.833, 45.0),
                          file="Gurbantonggut Desert.xlsx", blocks=[
        ("Sheet1", 0, "WL", 1, {
            ("Cyanobacteria", "Dry"): ["Dry cyanobacteria"], ("Cyanobacteria", "Wet"): ["Wet cyanobacteria"],
            ("Lichen", "Dry"): ["Dry lichens"], ("Lichen", "Wet"): ["Wet lichens"],
            ("Moss", "Dry"): ["Dry mosses"], ("Moss", "Wet"): ["Wet mosses"],
            ("Bare soil", "Dry"): ["Dry sands"], ("Bare soil", "Wet"): ["Wet sands"]}),
    ]),
    "Wajir": dict(full="Wajir County, Kenya", lonlat=(40.214, 1.531),
                  file="Wajir County Kenya.xlsx", blocks=[
        ("kenya_smoothed_wide", 1, 0, 1, {("Cyanobacteria", "Dry"): KENYA_B("D"), ("Cyanobacteria", "Wet"): KENYA_B("W"),
                                          ("Bare soil", "Dry"): KENYA_K("D"), ("Bare soil", "Wet"): KENYA_K("W")}, True),
    ]),
}
# Replicate spectra outside the workbooks, {site: {(class, cond): [csv files]}}; all columns but "wavelength"
# are replicates and override the workbook class mean so mean and SD share one source. Left empty: the Colorado
# Plateau workbook pools spectra from several Colorado Plateau locations, not only Castle Valley, so the Castle
# Valley CSVs in 1_data/0_field_spectra must not stand in for it (their DCY/LCY/moss means happen to match the
# workbook to r > 0.998, their lichen does not). Colorado, Namib, Soebatsfontein and Gurbantünggüt therefore
# show class means only (Namib's replicate sheets are a wetting time series whose steps do not equal the
# published dry/wet means).
CSV_DIR = os.path.join(os.path.dirname(f2.DATA_DIR), "0_field_spectra")
CSV_REPS = {}
SPAIN_MEANS = {"Las Amoladeras": ("VIP_scores", 8, "Label", 0.01, {   # workbook class means, for verify_spain
                   ("Cyanobacteria", "Dry"): 22, ("Lichen", "Dry"): 28, ("Moss", "Dry"): 34, ("Bare soil", "Dry"): 40,
                   ("Cyanobacteria", "Wet"): 49, ("Lichen", "Wet"): 55, ("Moss", "Wet"): 61, ("Bare soil", "Wet"): 67}),
               "El Cautivo": ("VIP_CAUT", 8, "Label", 0.01, {
                   ("Cyanobacteria", "Dry"): 20, ("Lichen", "Dry"): 26, ("Bare soil", "Dry"): 32,
                   ("Cyanobacteria", "Wet"): 41, ("Lichen", "Wet"): 47, ("Bare soil", "Wet"): 53})}

# Karnieli et al. Table 4: dominant wavelengths (nm) retrieved from the PLS-R VIP scores, per site
REGIONS = ["NUV", "Blue", "Red", "NIR", "SWIR2"]
TABLE4 = {
    "Negev": (350, 468, 683, 1106, 2058),
    "Namib": (368, 409, 696, 1025, 2055),
    "Colorado Plateau": (351, 430, 684, 759, 2053),
    "Las Amoladeras": (373, 458, 695, 804, 2055),
    "El Cautivo": (374, 487, 692, 753, 2057),
    "Soebatsfontein": (365, 477, 692, 759, 2055),
    "Gurbantünggüt": (356, 469, 685, 1005, 2057),
    "Wajir": (350, 496, 688, 1001, 2055),
}
REGION_STAT = "range"   # shaded region across sites: "range" = min-max of Table 4, "sd" = mean ± 1 SD
MIN_BAND_WIDTH = 20     # nm; the red and SWIR2 ranges are only 13 and 5 nm wide - widened so the shading shows

# spectral windows used to compute the indices from the field spectra (nm); Blue-SWIR2 follow the Landsat-9
# windows listed in Karnieli et al. Table 4, SWIR2 narrowed to the cellulose feature they highlight (Fig. 4)
BANDS = {"NUV": (350, 400), "Blue": (450, 510), "Green": (530, 590), "Red": (640, 670), "NIR": (850, 880),
         "SWIR1": (1570, 1650), "SWIR2": (2050, 2200)}
# formulas as mathtext (Arial via NATURE_RC) so BOMI gets a real radical with its vinculum
INDICES = [("CI", "cyanobacteria crust index", r"$1 - (\mathrm{Red} - \mathrm{Blue})/(\mathrm{Red} + \mathrm{Blue})$"),
           ("SCI", "scytonemin index", r"$1 - (\mathrm{Green} - \mathrm{NUV})/(\mathrm{Green} + \mathrm{NUV})$"),
           ("BOMI", "biocrust organic matter index", r"$1 - \sqrt{(\mathrm{Blue}^2 + \mathrm{Red}^2 + \mathrm{NIR}^2)/3}$"),
           ("SCAI", "satellite cellulose absorption index", r"$(\mathrm{SWIR1} - \mathrm{SWIR2})/(\mathrm{SWIR1} + \mathrm{SWIR2})$")]
B_CONDITIONS = ("Dry", "Wet")   # spectra entering b's boxes: dry and wet pooled (one value each per site)
# b: one symbol per site (Okabe-Ito colour + marker shape, so sites stay apart in greyscale too); marker area
SITE_STYLE = {"Negev": ("#0072B2", "o", 9), "Namib": ("#D55E00", "s", 8), "Colorado Plateau": ("#009E73", "^", 10),
              "Las Amoladeras": ("#CC79A7", "D", 7), "El Cautivo": ("#E69F00", "v", 10),
              "Soebatsfontein": ("#56B4E9", "P", 11), "Gurbantünggüt": ("#F0E442", "X", 11), "Wajir": ("#000000", "*", 16)}


# ================================================================ data
def column(df, c):
    return df.iloc[:, c] if isinstance(c, int) else df[c]


def on_grid(wl, arr, reps):
    """Mean of arr's columns on WL; SD across them (and n) only if the columns are replicate spectra."""
    empty = [c for c, n in arr.notna().sum().items() if n == 0]
    assert not empty, f"all-NaN spectrum column(s): {empty}"
    ok = wl.notna() & arr.notna().any(axis=1)          # mean/SD skip NaN cells (e.g. removed water bands)
    mean = np.interp(WL, wl[ok], arr[ok].mean(axis=1))
    sd = np.interp(WL, wl[ok], arr[ok].std(axis=1, ddof=1)) if reps and arr.shape[1] > 1 else None
    return {"mean": mean, "sd": sd, "n": arr.shape[1] if reps else 1}


def load_library():
    """{site: {(class, cond): {"mean", "sd", "n"}}} - class mean per site; sd only where replicates exist."""
    lib = {}
    for site, spec in SITES.items():
        lib[site] = {}
        for block in spec["blocks"]:
            sheet, header, wl_col, scale, classes = block[:5]
            reps = block[5] if len(block) > 5 else False
            df = pd.read_excel(os.path.join(SPEC_DIR, spec["file"]), sheet_name=sheet, header=header)
            wl = pd.to_numeric(column(df, wl_col), errors="coerce")
            for key, cols in classes.items():
                arr = pd.concat([pd.to_numeric(column(df, c), errors="coerce") for c in cols], axis=1) * scale
                lib[site][key] = on_grid(wl, arr, reps)
        for key, files in CSV_REPS.get(site, {}).items():
            frames = [pd.read_csv(os.path.join(CSV_DIR, f)) for f in files]
            arr = pd.concat([f.drop(columns="wavelength") for f in frames], axis=1).apply(pd.to_numeric)
            lib[site][key] = on_grid(frames[0]["wavelength"], arr, True)
    return lib


def verify_spain(lib):
    """The mean of the four plot means must reproduce the workbook's own class-mean column."""
    for site, (sheet, header, wl_col, scale, cols) in SPAIN_MEANS.items():
        df = pd.read_excel(os.path.join(SPEC_DIR, SITES[site]["file"]), sheet_name=sheet, header=header)
        wl = pd.to_numeric(column(df, wl_col), errors="coerce")
        for key, c in cols.items():
            ref = on_grid(wl, pd.to_numeric(column(df, c), errors="coerce").to_frame() * scale, False)["mean"]
            rmse = np.sqrt(np.nanmean((lib[site][key]["mean"] - ref) ** 2))
            flag = "" if rmse < 1e-3 else "   <-- differs from the workbook class mean"
            print(f"  {site:15s} {key[0]:13s} {key[1]}: replicate mean vs workbook mean RMSE = {rmse:.5f}{flag}")


def band_mean(refl, name):
    lo, hi = BANDS[name]
    return float(np.nanmean(refl[(WL >= lo) & (WL <= hi)]))


def crust_indices(refl):
    b = {name: band_mean(refl, name) for name in BANDS}
    return {
        "CI": 1 - (b["Red"] - b["Blue"]) / (b["Red"] + b["Blue"]),
        "SCI": 1 - (b["Green"] - b["NUV"]) / (b["Green"] + b["NUV"]),
        "BOMI": 1 - np.sqrt((b["Blue"] ** 2 + b["Red"] ** 2 + b["NIR"] ** 2) / 3),
        "SCAI": (b["SWIR1"] - b["SWIR2"]) / (b["SWIR1"] + b["SWIR2"]),
    }


def region_ranges():
    """{region: (lo, hi)} across the eight sites (REGION_STAT), widened to MIN_BAND_WIDTH."""
    out = {}
    for k, region in enumerate(REGIONS):
        vals = np.array([v[k] for v in TABLE4.values()], dtype=float)
        if REGION_STAT == "range":
            lo, hi = vals.min(), vals.max()
        else:
            lo, hi = vals.mean() - vals.std(ddof=1), vals.mean() + vals.std(ddof=1)
        pad = max(0, (MIN_BAND_WIDTH - (hi - lo)) / 2)
        out[region] = (lo - pad, hi + pad)
        print(f"  {region:5s} Table 4 {vals.min():.0f}-{vals.max():.0f} nm (mean {vals.mean():.0f}, SD {vals.std(ddof=1):.0f})"
              f" -> shaded {lo - pad:.0f}-{hi + pad:.0f} nm")
    return out


# ================================================================ drawing helpers
axes_in = f2.axes_in
heading = f2.heading
clean = f2.clean

REGION_COLOR = "#f6c66d"     # preserved regions in the strip above each panel (amber)
SD_ALPHA = 0.15              # ± 1 SD band around a class mean (class colour)
BOX_ALPHA = 0.30             # b: box fill in the class colour
B_TITLE_PAD = 15             # b: title pad (points) leaves room for the formula line (BOMI's radical is tall)
B_FORMULA_H = 0.18           # inches reserved for that formula line in the layout
A_LW = 1.2                   # a: spectra line width (dry and wet)
KEY_FONT = 6                 # key rows under headings a and b (pt)
FORMULA_FONT = 6             # b: formula line under each title (pt)
SITE_TICK = dict(color="black", lw=0.7)   # the site's dominant wavelengths: short ticks through the strip
STRIP_H, STRIP_GAP = 0.07, 0.015         # strip height and its gap to the spectra axes (inches)


def water_bands(ax):
    for lo, hi in BAD_BANDS:
        ax.axvspan(lo, hi, color="#f0f0f0", lw=0, zorder=0)


def spectra_axes(ax, xlabel=True):
    ax.set_xlim(*XLIM)
    ax.set_xticks([500, 1000, 1500, 2000, 2500])
    if xlabel:
        ax.set_xlabel("Wavelength (nm)")
    clean(ax)


# ================================================================ panels
A_YMAX = 0.8                 # common to all sites (class means peak at 0.64, + SD bands)


def region_strip(ax, site, ranges, title):
    """Ruler above a site's spectra: the cross-site region ranges as amber segments (identical in every panel)
    and the site's own Table 4 wavelengths as short black ticks poking through them."""
    for lo, hi in ranges.values():
        ax.axvspan(lo, hi, ymin=0.18, ymax=0.82, color=REGION_COLOR, lw=0, zorder=1)
    for wl0 in TABLE4[site]:
        ax.plot([wl0, wl0], [0, 1], zorder=2, solid_capstyle="butt", **SITE_TICK)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, loc="left", pad=2)


def panel_site(ax, site, spectra, ranges, k, ylabel=False, xlabel=False):
    """One site: class-mean dry (solid) and wet (dashed) spectra with a ± 1 SD band where the source has
    replicate spectra; the regions and the site's wavelengths are drawn in the strip above (region_strip)."""
    water_bands(ax)
    n_lines, ns = 0, []
    for cls in CLASSES:
        for cond, ls in COND_STYLES.items():
            if (cls, cond) in spectra:
                s = spectra[(cls, cond)]
                mean = f2.mask_bad_bands(WL, s["mean"])
                if s["sd"] is not None:
                    sd = f2.mask_bad_bands(WL, s["sd"])
                    ax.fill_between(WL, mean - sd, mean + sd, color=CLASS_COLORS[cls], alpha=SD_ALPHA, lw=0, zorder=2)
                ax.plot(WL, mean, color=CLASS_COLORS[cls], lw=A_LW, ls=ls, zorder=3)
                n_lines += 1
                ns.append(f"{cls[:5]}/{cond[0]} n={s['n']}")
    if not any(cond == "Wet" for _, cond in spectra):
        ax.text(0.98, 0.04, "dry spectra only", transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
                color="#555555", fontstyle="italic")
    spectra_axes(ax, xlabel=xlabel)
    ax.set_ylim(0, A_YMAX)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    if ylabel:
        ax.set_ylabel("Reflectance")
    print(f"  a.{k + 1} {site:17s} {n_lines} spectra; Table 4 nm = {TABLE4[site]}; " + ", ".join(ns))


def panel_indices(axes, lib):
    """Per index: one neutral box per biocrust type across sites (B_CONDITIONS spectra), one symbol per site."""
    values = {}
    for site, spectra in lib.items():
        for (cls, cond), s in spectra.items():
            if cond not in B_CONDITIONS:
                continue
            for name, v in crust_indices(s["mean"]).items():
                values.setdefault((name, cls), []).append((site, v))
    sites = list(SITES)
    for k, (ax, (name, _, formula)) in enumerate(zip(axes, INDICES)):
        for i, cls in enumerate(CLASSES):
            pts = values.get((name, cls), [])
            vals = [v for _, v in pts]
            color = CLASS_COLORS[cls]                      # box in the type's colour of panel a
            ax.boxplot([vals], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=matplotlib.colors.to_rgba(color, BOX_ALPHA), edgecolor=color, lw=0.6),
                       whiskerprops=dict(lw=0.6, color=color), capprops=dict(lw=0.6, color=color),
                       medianprops=dict(lw=1.0, color="black"), zorder=2)
            for site, v in pts:
                color, marker, size = SITE_STYLE[site]
                jitter = (sites.index(site) - (len(sites) - 1) / 2) * 0.045
                ax.scatter(i + jitter, v, s=size, marker=marker, color=color, edgecolor="black", linewidth=0.3,
                           zorder=4)
            ax.text(i, -0.03, str(len(vals)), transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=4.5, color="#555555")
        ax.text(-0.6, -0.03, "n", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=4.5,
                color="#555555", fontstyle="italic")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(["Cyano-\nbacteria", "Lichen", "Moss", "Bare\nsoil"])
        ax.tick_params(axis="x", pad=8, length=0)
        ax.set_xlim(-0.6, len(CLASSES) - 0.4)
        ax.set_title(f"(b.{k + 1}) {name}", loc="left", pad=B_TITLE_PAD)
        ax.text(0, 1.03, formula, transform=ax.transAxes, ha="left", va="bottom", fontsize=FORMULA_FONT,
                color="#333333")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        if k == 0:
            ax.set_ylabel("Index value")
        clean(ax)
    for name, _, _ in INDICES:
        print(f"  b: {name:4s} " + " | ".join(
            f"{cls}: " + ", ".join(f"{s} {v:.3f}" for s, v in values.get((name, cls), [])) for cls in CLASSES))


def site_key(fig, x, y):
    """One row of site symbols (b), lower-left corner at (x, y) inches."""
    W, H = fig.get_size_inches()
    handles = [Line2D([], [], ls="none", marker=m, markersize=np.sqrt(s) * 1.15, markerfacecolor=c,
                      markeredgecolor="black", markeredgewidth=0.3) for c, m, s in SITE_STYLE.values()]
    fig.legend(handles=handles, labels=list(SITE_STYLE), loc="lower left", bbox_to_anchor=(x / W, y / H),
               ncol=len(handles), frameon=False, handlelength=1.2, columnspacing=1.0, handletextpad=0.4,
               borderaxespad=0, borderpad=0, fontsize=KEY_FONT)


# ================================================================ figure
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(NATURE_RC)
    lib = load_library()
    verify_spain(lib)
    ranges = region_ranges()

    LEFT, RIGHT = 0.46, 0.05
    HEAD = 0.30                 # panel letter + heading + sub-panel title
    KEY_ROW = 0.27              # a: two key rows between the heading and the first row of panels
    STRIP = STRIP_H + STRIP_GAP  # a: region strip above each spectra axes
    ROW_GAP = 0.17 + STRIP      # second row's sub-panel titles + strips (the top row has no x tick labels)
    XLAB = 0.30                 # x tick labels + "Wavelength (nm)"
    a_h, b_h, b_xlab = 0.95, 1.00, 0.44
    gap = 0.30                  # column gap: just the next panel's y tick labels
    full_w = FIG_W - RIGHT - LEFT
    a_w = (full_w - 3 * gap) / 4
    b_gap = 0.22
    b_w = (full_w - 3 * b_gap) / 4
    B_KEY = 0.15                # b: site-symbol key row under the heading
    B_HEAD = HEAD + B_FORMULA_H + B_KEY   # heading + key row + title + formula line
    H = HEAD + KEY_ROW + STRIP + a_h + ROW_GAP + a_h + XLAB + B_HEAD + b_h + b_xlab + 0.02
    fig = plt.figure(figsize=(FIG_W, H))

    # a: one panel per site, 2 rows x 4 columns, each with its region strip (and title) on top
    top = H - HEAD - KEY_ROW - STRIP
    heading(fig, 0.12, top + STRIP + KEY_ROW + 0.16, "a",
            "Biocrust spectral library of eight drylands: dry and wet reflectance")
    for k, site in enumerate(SITES):
        row, col = divmod(k, 4)
        bottom = top - a_h - row * (a_h + ROW_GAP)
        x = LEFT + col * (a_w + gap)
        ax = axes_in(fig, x, bottom, a_w, a_h)
        panel_site(ax, site, lib[site], ranges, k, ylabel=(col == 0), xlabel=(row == 1))
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)
        region_strip(axes_in(fig, x, bottom + a_h + STRIP_GAP, a_w, STRIP_H), site, ranges,
                     f"(a.{k + 1}) {SITES[site]['full']}")
    # two key rows under the heading: (1) classes, dry/wet, SD band; (2) the strip's regions and site ticks
    row1 = [Line2D([], [], color=CLASS_COLORS[c], lw=1.2) for c in CLASSES]
    row1 += [Line2D([], [], color="black", lw=1.0, ls=ls) for ls in COND_STYLES.values()]
    row1 += [Patch(facecolor="#555555", alpha=SD_ALPHA + 0.1, edgecolor="none")]
    labels1 = CLASSES + list(COND_STYLES) + ["Mean ± 1 SD (where replicate spectra exist)"]
    row2 = [Patch(facecolor=REGION_COLOR, edgecolor="none"),
            Line2D([], [], ls="none", marker="|", markersize=5, markeredgewidth=SITE_TICK["lw"], color="black")]
    labels2 = ["Spectral regions preserved across sites (left to right: NUV, blue, red, NIR, SWIR2)",
               "Dominant wavelengths of the site"]
    for handles, labels, y in ((row1, labels1, top + STRIP + 0.28), (row2, labels2, top + STRIP + 0.15)):
        fig.legend(handles=handles, labels=labels, loc="lower left", bbox_to_anchor=(LEFT / FIG_W, y / H),
                   ncol=len(handles), frameon=False, handlelength=1.8, columnspacing=1.0, handletextpad=0.5,
                   borderaxespad=0, borderpad=0, fontsize=KEY_FONT)

    # b: indices from the dry spectra, one symbol per site
    top = top - 2 * a_h - ROW_GAP - XLAB - B_HEAD
    heading(fig, 0.12, top + 0.16 + B_FORMULA_H + B_KEY, "b",
            "Biocrust indices built from the preserved regions, by biocrust type across sites")
    site_key(fig, LEFT, top + 0.16 + B_FORMULA_H + 0.02)
    axes = [axes_in(fig, LEFT + k * (b_w + b_gap), top - b_h, b_w, b_h) for k in range(4)]
    panel_indices(axes, lib)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{OUT_NAME}.{ext}")
        fig.savefig(out, dpi=1000, facecolor="white")
        print("wrote", out)
    plt.close(fig)
    print(f"figure {FIG_W / MM:.0f} x {H / MM:.0f} mm")


if __name__ == "__main__":
    main()
