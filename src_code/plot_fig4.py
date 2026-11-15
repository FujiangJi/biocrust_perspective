"""Figure 4: airborne and spaceborne imaging spectrometers relevant to biocrust mapping (the former Table 1 as a
figure). Two stacked charts:

a  spectral coverage: one block per instrument over the atmospheric transmittance (ASTM G173-03 AM1.5 direct beam /
   extraterrestrial irradiance); on each block: instrument (bold) · launch · platform · agency · sampling · bands ·
   revisit. Blocks are coloured by tier (airborne / spaceborne / planned hollow / PACE OCI). The seven bands of the
   biocrust indices (NUV, blue, green, red, NIR, SWIR1, SWIR2) are tinted amber across the panel, and the matrix at
   the right shows for each instrument which index can be computed from its coverage (filled = all bands inside the
   contiguous range; ring = covered by PACE OCI's discrete bands; open = not covered). PACE OCI = contiguous
   340-895 nm block plus seven discrete SWIR bands. Strip below: wavelength regions and the biocrust properties they
   carry.
b  mission timeline with a broken year axis: first operation or launch (x) against ground sampling distance (y,
   log; a bar where it depends on flight altitude), marker area = swath, colour = tier.

The qualitative strengths of the table's last column (heritage archive, anchor sensor, ...) go to the caption.
Values are those of Table 1 of the manuscript (v7), entered in SENSORS below; index band windows follow plot_fig3
(NUV taken as 380-400 nm, the EMIT window of Karnieli et al.).
Sized for print: 183 mm wide, Arial 5-8 pt, 1000 dpi PNG plus vector PDF with editable text.
Run with the Fujiang env (needs pvlib for the ASTM G173 spectrum):
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_fig4.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

import plot_fig2 as f2
from plot_fig2 import axes_in, heading, clean

FIG_DIR = f2.FIG_DIR
OUT_NAME = "Fig4"
MM = 1 / 25.4
FIG_W = 183 * MM
NATURE_RC = f2.NATURE_RC

# ---------------------------------------------------------------- data (Table 1)
# spectral: (lo, hi) nm segments; gsd (m) / swath (km): (lo, hi), equal values = a point, None = altitude-dependent
# year: first operation or launch (planned missions ~2028); label: (dx, dy) points and alignment of the name in b
PACE_SWIR = [940, 1038, 1250, 1378, 1615, 2130, 2260]          # OCI discrete SWIR bands (nm)
SENSORS = [
    dict(name="AVIRIS-Classic", launch="1987–", year=1987, platform="ER-2 / Twin Otter · NASA JPL", tier="air",
         spectral=[(380, 2510)], sampling="~10 nm · 224 bands", gsd=(2, 20), swath=(2, 11), revisit="on demand",
         label=(5, 0, "left", "center")),
    dict(name="AVIRIS-NG", launch="2014–", year=2014, platform="Twin Otter / King Air · NASA JPL", tier="air",
         spectral=[(380, 2510)], sampling="5 nm · 425 bands", gsd=(0.3, 20), swath=(1, 12), revisit="on demand",
         label=(5, 0, "left", "center")),
    dict(name="AVIRIS-3", launch="2023–", year=2023, platform="various aircraft · NASA JPL", tier="air",
         spectral=[(380, 2500)], sampling="~7.5 nm · ~285 bands", gsd=(0.3, 15), swath=None,
         revisit="on demand", label=(-5, 0, "right", "center")),
    dict(name="AVIRIS-5", launch="2025–", year=2025, platform="ER-2 / B-200 / Gulfstream · NASA JPL", tier="air",
         spectral=[(380, 2500)], sampling="5 nm · 425 bands", gsd=(0.5, 13), swath=None, revisit="on demand",
         label=(5, 0, "left", "center")),
    dict(name="NEON AOP", launch="2013–", year=2013, platform="Twin Otter · NSF NEON", tier="air",
         spectral=[(380, 2510)], sampling="~5 nm · 426 bands", gsd=(1, 1), swath=(1, 1),
         revisit="annual revisit, US sites", label=(-6, 0, "right", "center")),
    dict(name="EnMAP", launch="Apr 2022", year=2022.3, platform="satellite · DLR", tier="space",
         spectral=[(420, 2450)], sampling="6.5 / 10 nm · 224 bands", gsd=(30, 30), swath=(30, 30),
         revisit="revisit 27 d, ~4 d pointed", label=(-6, 0, "right", "center")),
    dict(name="PRISMA", launch="Mar 2019", year=2019.2, platform="satellite · ASI", tier="space",
         spectral=[(400, 2500)], sampling="~10 nm · 239 bands", gsd=(30, 30), gsd_text="30 m, 5 m PAN", pan=5, swath=(30, 30),
         revisit="revisit ~29 d, ~7 d pointed", label=(0, 12, "center", "bottom")),
    dict(name="DESIS", launch="2018", year=2018.5, platform="ISS / MUSES · DLR", tier="space",
         spectral=[(400, 1000)], sampling="2.55 nm · 235 bands", gsd=(30, 30), swath=(30, 30),
         revisit="irregular revisit (ISS)", label=(-7, 0, "right", "center")),
    dict(name="HISUI", launch="2019", year=2019.9, platform="ISS · METI", tier="space",
         spectral=[(400, 2500)], sampling="10 / 12.5 nm · 185 bands", gsd=(20, 30), gsd_text="20 × 30 m", swath=(20, 20),
         revisit="irregular revisit (ISS)", label=(0, -11, "center", "top")),
    dict(name="Tanager-1", launch="Aug 2024", year=2024.6, platform="satellite · Planet / Carbon Mapper",
         tier="space", spectral=[(400, 2500)], sampling="~5 nm", gsd=(30, 30), gsd_text="~30 m", swath=(18, 18), revisit="taskable", label=(6, 0, "left", "center")),
    dict(name="EMIT", launch="Jul 2022", year=2022.5, platform="ISS · NASA JPL", tier="space",
         spectral=[(380, 2500)], sampling="7.4 nm · 285 bands", gsd=(60, 60), swath=(75, 75),
         revisit="revisit of days over arid zones (ISS)", label=(8, 0, "left", "center")),
    dict(name="SBG / EAGLE", launch="~2028+", year=2028, platform="Earth System Observatory · NASA",
         tier="planned", spectral=[(380, 2500)], sampling="~10 nm", gsd=(30, 30), swath=(185, 185),
         revisit="revisit ~16 d", label=(0, 12, "center", "bottom")),
    dict(name="CHIME", launch="~2028+", year=2028.8, platform="Copernicus · ESA / EU", tier="planned",
         spectral=[(400, 2500)], sampling="~10 nm", gsd=(30, 30), swath=(130, 130),
         revisit="revisit ~10–12.5 d (2 satellites)", label=(8, 0, "left", "center")),
    dict(name="PACE OCI", launch="Feb 2024", year=2024.1, platform="satellite · NASA", tier="pace",
         spectral=[(340, 895)] + [(b - 7, b + 7) for b in PACE_SWIR], sampling="5 nm VNIR + 7 SWIR bands",
         gsd=(1000, 1000), swath=(2600, 2600), revisit="revisit ~2 d, global",
         label=(12, 0, "left", "center")),
]
TIER_COLORS = {"air": "#2a9d8f", "space": "#4a5fa5", "planned": "#4a5fa5", "pace": "#e8a33d"}
TIER_NAMES = {"air": "Airborne", "space": "Spaceborne", "planned": "Planned", "pace": "Global"}
LEGEND = [("air", False, "Airborne: fine-scale reference"),
          ("space", False, "Spaceborne: landscape to regional"),
          ("planned", True, "Planned mission"),
          ("pace", False, "Spaceborne: global, frequent revisit (PACE OCI)")]
REGIONS = [  # wavelength regions and what they carry for biocrusts (strip under a)
    (380, 700, "Visible", "Pigments: scytonemin,\ncarotenoids, phycobilins,\nchlorophyll a", "#cfe1f3"),
    (700, 1300, "Near infrared", "Red edge, cell structure,\nhydration", "#fbe0bf"),
    (1300, 2500, "Shortwave infrared", "Water, cellulose, lignin,\nclay minerals", "#e2e2e2"),
]
# bands of the biocrust indices (plot_fig3.BANDS; NUV = the 380-400 nm EMIT window of Karnieli et al.)
BANDS = {"NUV": (380, 400), "Blue": (450, 510), "Green": (530, 590), "Red": (640, 670), "NIR": (850, 880),
         "SWIR1": (1570, 1650), "SWIR2": (2050, 2200)}
INDEX_BANDS = {"CI": ("Blue", "Red"), "SCI": ("NUV", "Green"), "BOMI": ("Blue", "Red", "NIR"),
               "SCAI": ("SWIR1", "SWIR2")}
BAND_COLOR = "#e8a33d"
INK, GREY = "#1d2a3a", "#6f7781"
TRANS_FILL, TRANS_LINE = "#dce3ec", "#7f93b0"
FONT = 5
ROW, GROUP_GAP, HEADER = 1.0, 0.5, 1.2
BAR_H, BAR_DY = 0.42, -0.24      # block height and centre offset; the text line sits above the block
TEXT_DY = 0.06                    # bottom of the text line above the row centre


def row_positions():
    """y of each sensor in a (top row = 0, downwards) and the dashed separators between tiers."""
    ys, seps, y, prev = [], [], 0.0, None
    for s in SENSORS:
        if prev is not None and s["tier"] != prev:
            seps.append(-(y - ROW + (ROW + GROUP_GAP) / 2))
            y += GROUP_GAP
        ys.append(-y)
        y += ROW
        prev = s["tier"]
    return ys, seps


def transmittance():
    """Atmospheric transmittance proxy: ASTM G173-03 AM1.5 direct(+circumsolar) / extraterrestrial irradiance."""
    from pvlib import spectrum
    ref = spectrum.get_reference_spectra(standard="ASTM G173-03")
    T = (ref["direct"] / ref["extraterrestrial"]).clip(0, 1).loc[300:2600]
    return T.index.to_numpy(float), T.to_numpy(float)


def band_coverage(s, band):
    """'full' if a contiguous segment holds the whole band window, 'discrete' if a narrow band falls inside it,
    else 'none'."""
    lo, hi = BANDS[band]
    for a, b in s["spectral"]:
        if a <= lo and b >= hi:
            return "full"
    for a, b in s["spectral"]:
        if b - a < 30 and lo <= (a + b) / 2 <= hi:
            return "discrete"
    return "none"


def index_coverage(s, index):
    cov = [band_coverage(s, b) for b in INDEX_BANDS[index]]
    if all(c == "full" for c in cov):
        return "full"
    if all(c != "none" for c in cov):
        return "discrete"
    return "none"


# ---------------------------------------------------------------- a
NAME_FONT = 6            # a: instrument names (bold, tier colour); details stay at FONT
NAME_COLORS = {"air": "#1f7f73", "space": "#3d508f", "planned": "#3d508f", "pace": "#b8781a"}   # darker for text


def block_label(ax, x, y, name, rest, name_color):
    """Bold instrument name (tier colour) followed by the regular-weight details, one line above the block."""
    effects = [pe.withStroke(linewidth=1.0, foreground="white")]
    t = ax.text(x, y, name, ha="left", va="bottom", fontsize=NAME_FONT, fontweight="bold", color=name_color,
                zorder=4, path_effects=effects)
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    x_end = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x_end + 12, y + 0.02, rest, ha="left", va="bottom", fontsize=FONT, color=INK, zorder=4,
            path_effects=effects)


def panel_a(ax, ys, seps, in_per_nm):
    y_top, y_bot = ys[0] + 0.5 + HEADER, ys[-1] - 0.5
    wl, T = transmittance()
    ax.fill_between(wl, y_bot, y_bot + T * (y_top - y_bot), color=TRANS_FILL, lw=0, zorder=0)
    ax.plot(wl, y_bot + T * (y_top - y_bot), color=TRANS_LINE, lw=0.5, zorder=0.5)
    for name, (lo, hi) in BANDS.items():                                   # index bands, tinted
        ax.axvspan(lo, hi, color=BAND_COLOR, alpha=0.22, lw=0, zorder=0.6)
        ax.text((lo + hi) / 2, y_top - 0.04, name, ha="center", va="top", fontsize=4.5, color="#b07a1f",
                rotation=0 if hi - lo > 45 else 90)
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks([y_bot + f * (y_top - y_bot) for f in (0, 0.5, 1)])
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_ylabel("Atmospheric transmittance (%)")
    for x in (1380, 1870):
        ax.text(x, y_top - 0.05, "H$_2$O", ha="center", va="top", fontsize=FONT, color=GREY)
    for s, y in zip(SENSORS, ys):
        color, planned = TIER_COLORS[s["tier"]], s["tier"] == "planned"
        for lo, hi in s["spectral"]:
            ax.barh(y + BAR_DY, hi - lo, left=lo, height=BAR_H, facecolor="white" if planned else color,
                    edgecolor=color, linewidth=0.7 if planned else 0, zorder=2)
        lo = s["spectral"][0][0]
        rest = f"({s['launch']}) · {s['platform']} · {s['sampling']} · {s['revisit']}"
        block_label(ax, lo, y + TEXT_DY, s["name"], rest, NAME_COLORS[s["tier"]])
    for y in seps:
        ax.axhline(y, color="#c8ced6", lw=0.5, ls=(0, (3, 2)), zorder=1)
    for tier, name in TIER_NAMES.items():                                  # tier names at the right
        rows = [y for s, y in zip(SENSORS, ys) if s["tier"] == tier]
        ax.text(2545, np.mean(rows), name, ha="left", va="center", fontsize=FONT, color=GREY, rotation=90,
                clip_on=False)
    ax.set_xlim(330, 2530)
    ax.set_xticks([500, 1000, 1500, 2000, 2500])
    ax.set_xticklabels(["500", "1,000", "1,500", "2,000", "2,500"])
    ax.set_xlabel("Wavelength (nm)")
    clean(ax)
    return y_bot, y_top


def region_strip(ax):
    for lo, hi, name, feats, color in REGIONS:
        ax.add_patch(plt.Rectangle((lo, 0.72), hi - lo, 0.28, facecolor=color, edgecolor="none"))
        ax.text((lo + hi) / 2, 0.86, name, ha="center", va="center", fontsize=FONT, fontweight="bold", color=INK)
        ax.text((lo + hi) / 2, 0.64, feats, ha="center", va="top", fontsize=FONT, color=GREY, linespacing=1.05)
    ax.set_xlim(330, 2530)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ---------------------------------------------------------------- b
def gsd_text(s):
    """Ground sampling distance for the label in b."""
    if "gsd_text" in s:
        return s["gsd_text"]
    lo, hi = s["gsd"]
    fmt = lambda v: f"{v:g}"
    if lo == hi:
        return "~1 km" if lo >= 1000 else f"{fmt(lo)} m"
    return f"{fmt(lo)}–{fmt(hi)} m"


LABEL_PITCH = 6.5        # points between label lines in b


def point_label(ax, xy, lines, offset, ha, va, leader_r):
    """Stacked label lines beside a point: [(text, size, colour), ...]; the line nearest the point carries the
    leader when the label sits above or below it."""
    dx, dy = offset
    n = len(lines)
    if va == "center":
        offs = [dy + LABEL_PITCH * ((n - 1) / 2 - i) for i in range(n)]
    elif va == "bottom":
        offs = [dy + LABEL_PITCH * (n - 1 - i) for i in range(n)]
    else:
        offs = [dy - LABEL_PITCH * i for i in range(n)]
    nearest = n - 1 if va == "bottom" else 0
    arts = []
    for i, (text, size, color) in enumerate(lines):
        arrow = (dict(arrowstyle="-", color="#9aa4b0", lw=0.45, shrinkA=1.5, shrinkB=leader_r + 1.5)
                 if (i == nearest and dy) else None)
        arts.append(ax.annotate(text, xy, xytext=(dx, offs[i]), textcoords="offset points", ha=ha, va=va,
                                fontsize=size, color=color, zorder=5, arrowprops=arrow))
    if ha != "center":                      # centre the lines on each other, keeping the block's anchored side
        fig = ax.figure
        r = fig.canvas.get_renderer()
        widths = [r.get_text_width_height_descent(a.get_text(), a.get_fontproperties(), ismath=False)[0]
                  * 72 / fig.dpi for a in arts]                        # position-independent text widths
        for a, w, off in zip(arts, widths, offs):
            shift = (max(widths) - w) / 2
            a.xyann = (dx + shift if ha == "left" else dx - shift, off)


def swath_area(km):
    """Marker area (points^2) for a swath in km: 1 km -> 4, 10 -> 34, 100 -> 124, 1,000 -> 274, 2,600 -> 353."""
    return 4 + 30 * np.log10(km) ** 2


def swath_key(ax, x0, y0):
    """Nested circles with their bottoms aligned: marker area grows continuously with swath. (x0, y0) in axes
    coordinates is the common bottom; circles, leader lines and labels are offset in points, so they stay round."""
    fig = ax.figure
    kms = (10, 100, 1000)
    radii = [np.sqrt(swath_area(km)) / 2 for km in kms]              # points
    r_max = max(radii)
    for km, r in zip(kms, radii):
        ax.scatter([x0], [y0], s=swath_area(km), marker="o", facecolor="none", edgecolor="#8a93a0", linewidth=0.6,
                   transform=offset_copy(ax.transAxes, fig=fig, x=0, y=r, units="points"), clip_on=False, zorder=5)
        ax.annotate("", xy=(x0, y0), xytext=(x0, y0), textcoords=offset_copy(ax.transAxes, fig=fig, x=0, y=2 * r,
                    units="points"), xycoords=offset_copy(ax.transAxes, fig=fig, x=r_max + 3, y=2 * r, units="points"),
                    arrowprops=dict(arrowstyle="-", color="#b8bec8", lw=0.4, shrinkA=0, shrinkB=0), clip_on=False)
        ax.text(x0, y0, f"{km:,}", ha="left", va="center", fontsize=FONT, color=GREY, clip_on=False,
                transform=offset_copy(ax.transAxes, fig=fig, x=r_max + 5, y=2 * r, units="points"))
    ax.text(x0, y0, "Swath (km)", ha="center", va="bottom", fontsize=FONT, color=INK, fontweight="bold",
            clip_on=False, transform=offset_copy(ax.transAxes, fig=fig, x=0, y=2 * r_max + 6, units="points"))


def panel_b(ax1, ax2):
    """Timeline with a broken year axis: ax1 1985-1992 (AVIRIS-Classic), ax2 2010-2031 (everything else)."""
    ax2.axvspan(2018, 2031, color="#f1f4f8", lw=0, zorder=0)
    ax2.text(2018.3, 3300, "New generation of spaceborne imaging spectrometers", ha="left", va="center",
             fontsize=FONT, color=GREY, fontstyle="italic")
    for s in SENSORS:
        ax = ax1 if s["year"] < 2000 else ax2
        color, planned = TIER_COLORS[s["tier"]], s["tier"] == "planned"
        lo, hi = s["gsd"]
        km = s["swath"][0] if s["swath"] else 1
        if lo != hi:                                                       # airborne: altitude-dependent GSD
            ax.plot([s["year"]] * 2, [lo, hi], color=color, lw=2.2, solid_capstyle="round", zorder=3)
            ym = np.sqrt(lo * hi)
        else:
            ax.scatter(s["year"], lo, s=swath_area(km), marker="o", facecolor="white" if planned else color,
                       edgecolor=color if planned else "white", linewidth=0.8 if planned else 0.5, zorder=4)
            ym = lo
        dx, dy, ha, va = s["label"]
        r_pt = np.sqrt(swath_area(km)) / 2 if lo == hi else 1.5          # marker radius, to stop the leader short
        lines = [(s["name"], 5.5, INK), (f"({gsd_text(s)})", FONT, INK)]
        point_label(ax, (s["year"], ym), lines, (dx, dy), ha, va, r_pt)
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.set_ylim(0.15, 6000)
        clean(ax)
    ax1.set_yticks([1, 10, 100, 1000])
    ax1.set_yticklabels(["1", "10", "100", "1,000"])
    ax1.set_ylabel("Ground sampling distance (m)")
    ax1.set_xlim(1985.5, 1993.5)
    ax1.set_xticks([1987])
    ax2.set_xlim(2010.5, 2031)
    ax2.set_xticks([2015, 2020, 2025, 2030])
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax2.spines["left"].set_visible(False)
    ax2.patch.set_visible(False)                                           # labels may cross the break
    ax2.set_xlabel("First operation or launch")
    for ax, x in ((ax1, 1), (ax2, 0)):                                     # axis-break marks
        ax.plot([x - 0.012, x + 0.012], [-0.03, 0.03], transform=ax.transAxes, color="black", lw=0.5,
                clip_on=False)
    # tier legend (upper left, empty years) and swath key (lower right)
    handles = [Patch(facecolor="white" if hollow else TIER_COLORS[t], edgecolor=TIER_COLORS[t],
                     linewidth=0.8 if hollow else 0) for t, hollow, _ in LEGEND]
    ax1.legend(handles=handles, labels=[lab for _, _, lab in LEGEND], loc="upper left", bbox_to_anchor=(0.04, 1.0),
               frameon=False, handlelength=1.6, handleheight=0.8, labelspacing=0.35, handletextpad=0.5,
               borderaxespad=0.1)
    swath_key(ax2, 0.93, 0.06)


# ================================================================ figure
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(NATURE_RC)
    ys, seps = row_positions()

    LEFT, RIGHT = 0.42, 0.20        # RIGHT: rotated tier names beside a
    HEAD = 0.28
    XLAB, STRIP = 0.30, 0.34
    ROW_IN = 0.20
    a_h = (ys[0] + 0.5 + HEADER - (ys[-1] - 0.5)) * ROW_IN
    b_h = 1.60
    a_w = b_w = FIG_W - LEFT - RIGHT
    H = HEAD + a_h + XLAB + STRIP + HEAD + b_h + XLAB + 0.06
    fig = plt.figure(figsize=(FIG_W, H))

    b_bottom = XLAB + 0.02
    a_bottom = b_bottom + b_h + HEAD + STRIP + XLAB
    ax_a = axes_in(fig, LEFT, a_bottom, a_w, a_h)
    ax_s = axes_in(fig, LEFT, a_bottom - XLAB - STRIP + 0.02, a_w, STRIP)
    b1_w, b_gap = 1.00, 0.10
    ax_b1 = axes_in(fig, LEFT, b_bottom, b1_w, b_h)
    ax_b2 = axes_in(fig, LEFT + b1_w + b_gap, b_bottom, b_w - b1_w - b_gap, b_h)
    panel_a(ax_a, ys, seps, a_w / 2200)
    region_strip(ax_s)
    panel_b(ax_b1, ax_b2)
    heading(fig, 0.06, a_bottom + a_h + 0.08, "a", "Spectral coverage and biocrust index bands")
    # key for panel a's backgrounds, right-aligned on the heading line
    handles = [Patch(facecolor=BAND_COLOR, alpha=0.3, edgecolor="none"),
               Patch(facecolor=TRANS_FILL, edgecolor=TRANS_LINE, linewidth=0.5)]
    fig.legend(handles=handles, labels=["Biocrust-related spectral regions (index bands)",
                                        "Atmospheric transmittance"],
               loc="lower right", bbox_to_anchor=((FIG_W - RIGHT) / FIG_W, (a_bottom + a_h + 0.05) / H), ncol=2,
               frameon=False, handlelength=1.6, handleheight=0.8, columnspacing=1.4, handletextpad=0.5,
               borderaxespad=0, borderpad=0)
    heading(fig, 0.06, b_bottom + b_h + 0.08, "b", "Spatial resolution and swath by first operation or launch")

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{OUT_NAME}.{ext}")
        fig.savefig(out, dpi=1000, facecolor="white")
        print("wrote", out)
    plt.close(fig)
    print(f"figure {FIG_W / MM:.0f} x {H / MM:.0f} mm")


if __name__ == "__main__":
    main()
