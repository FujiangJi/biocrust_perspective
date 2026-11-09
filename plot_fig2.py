"""Figure 2: proof of concept - field hyperspectral reflectance resolves biocrust composition and its response to
experimental climate change at Castle Valley (Colorado Plateau). All data are from ref. 59 (GCB manuscript) but
are re-expressed here, so no panel repeats a figure of that paper.

a  (a.1) mean ± 1 SD contact-probe reflectance of the four biocrust functional types and bare soil, with the
   absorption features behind them annotated; (a.2) on the same wavelength axis, random-forest feature importance
   for each model target as a ridgeline (one filled curve per target, scaled to its maximum)
b  estimated vs measured cover per subplot (fine-tuned 1D-CNN): (b.1) late-, (b.2) early-successional biocrusts
c  response to the treatments, measured (solid, filled) and estimated (dashed, hollow), block mean ± SD:
   (c.1) successional stages, (c.2) functional types

Data (1_data/2_Fig2_data):
  ASD_All_Spectra_ContactProbe.csv               contact-probe spectra, rows block_treatment_class_replicate
  rf_separate_{4,6}class_refl_feature_imp.csv    RF feature importance per fold (4-class: stages; 6-class: BFTs)
  final_model_estimation_bilinear_{4,6}scheme.csv  measured and estimated cover per subplot (fine-tuned 1D-CNN)

Sized for print: 183 mm wide, Arial, 1000 dpi PNG plus vector PDF with editable text.
Run with the Fujiang env:
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_fig2.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DATA_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/0_src_code/1_data/2_Fig2_data"
FIG_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/figures"

PROBE_CSV = os.path.join(DATA_DIR, "ASD_All_Spectra_ContactProbe.csv")
FI_4 = os.path.join(DATA_DIR, "rf_separate_4class_refl_feature_imp.csv")
FI_6 = os.path.join(DATA_DIR, "rf_separate_6class_refl_feature_imp.csv")
EST_4 = os.path.join(DATA_DIR, "final_model_estimation_bilinear_4scheme.csv")
EST_6 = os.path.join(DATA_DIR, "final_model_estimation_bilinear_6scheme.csv")

MM = 1 / 25.4
FIG_W = 183 * MM

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 5.5,
    "axes.titlesize": 6.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 2,
    "axes.labelsize": 5.5,
    "axes.labelweight": "bold",
    "axes.labelpad": 1.5,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "xtick.minor.size": 1,
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
    "legend.fontsize": 5,
    "legend.title_fontsize": 5,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,
    "hatch.linewidth": 0.4,
    "mathtext.fontset": "custom",   # keep superscripts in Arial too
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial",
    "pdf.fonttype": 42,        # embed TrueType so text stays editable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
PANEL_LETTER_SIZE = 8
PANEL_TITLE_SIZE = 8

BAD_BANDS = [(1320, 1440), (1770, 1960)]   # water-absorption regions, masked in a
WL = np.arange(350, 2401)                  # common wavelength grid for a (350 nm so the scytonemin NUV band shows)

# model targets: colour (shared by all panels), feature-importance source, estimate source, contact-probe code
TARGETS = {
    "Late succ.":     dict(color="#4575b4", fi=(FI_4, "frac_late_successional"), est=(EST_4, "frac_late_successional")),
    "Early succ.":    dict(color="#fc8d59", fi=(FI_4, "frac_early_successional"), est=(EST_4, "frac_early_successional")),
    "LtCy":           dict(color="#e41a1c", fi=(FI_4, "frac_early_successional"), est=(EST_6, "frac_LtCy"), code="LCY"),
    "DkCy":           dict(color="#4daf4a", fi=(FI_6, "frac_DkCy"), est=(EST_6, "frac_DkCy"), code="DCY"),
    "Lichen":         dict(color="#00bfc4", fi=(FI_6, "frac_Lichen"), est=(EST_6, "frac_Lichen"), code="LCN"),
    "Moss":           dict(color="#984ea3", fi=(FI_6, "frac_Moss"), est=(EST_6, "frac_Moss"), code="MSS"),
    "Senescent veg.": dict(color="#a6761d", fi=(FI_4, "frac_Vegetation"), est=(EST_4, "frac_Vegetation")),
}
STAGES = ["Late succ.", "Early succ."]
BFT = ["LtCy", "DkCy", "Lichen", "Moss"]
RIDGES = ["Late succ.", "Early succ.", "DkCy", "Lichen", "Moss"]   # a.2, top to bottom

# a.1: absorption features behind the fingerprints: (label, centre nm or (lo, hi) band). The rotated labels hang
# from the top inside the axes; the y range leaves room for them above the SWIR curves (max ~0.6).
FEATURES = [                               # non-pigment features: grey band or dotted line, rotated label
    ("Red edge", (700, 750)),
    ("Cellulose", 2100),
    ("Lignin,\ncellulose", 2300),
]
# pigment absorption regions (nm) after Karnieli et al. (ref. 71, Fig. 1 and text): scytonemin NUV 370-400 nm
# (max 386), carotenoids ~450-550 nm (peak 516), phycobilins 520-670 nm (peak ~640), chlorophyll a 660-680 nm;
# the shaded windows are the dominant regions used for analysis in their Fig. 1. Colours follow that figure.
PIGMENTS = [
    ("Scytonemin", (370, 400), "#e7298a"),
    ("Carotenoids", (500, 540), "#ff7f00"),
    ("Phycobilins", (610, 650), "#1f78b4"),
    ("Chlorophyll a", (660, 680), "#33a02c"),
]
PIGMENT_ALPHA = 0.30
A1_YMAX = 0.75
MIX_CSV = os.path.join(DATA_DIR, "measured_mixtures.csv")   # subplot (mixed) spectra, rows block_treatment_subplot
RIDGE_SUB = {"Late succ.": "(DkCy, Lichen,\nMoss)", "Early succ.": "(LtCy)"}   # under the a.2 row labels

TREAT_CODES = {"CC": "Control", "CW": "AltP", "LC": "Warmed", "LW": "AltP+Warmed"}
TREATS = ["Control", "AltP", "Warmed", "AltP+Warmed"]
TREAT_COLORS = {"Control": "#2c7bb6", "AltP": "#74add1", "Warmed": "#fdae61", "AltP+Warmed": "#d7191c"}   # b


# ================================================================ data
def spectra_by_code(csv, field):
    """{code: (n, len(WL)) reflectance}; code = the given underscore field of the row id."""
    df = pd.read_csv(csv)
    bands = [c for c in df.columns if c.replace(".", "", 1).isdigit()]   # the mixtures file also holds cover columns
    wl = np.array([float(c) for c in bands])
    codes = df["full"].str.split("_").str[field].to_numpy()
    refl = df[bands].to_numpy(dtype=float)
    return {code: np.vstack([np.interp(WL, wl, r) for r in refl[codes == code]]) for code in np.unique(codes)}


def load_probe():
    """{class code: (n_records, len(WL)) reflectance} from the contact-probe CSV."""
    return spectra_by_code(PROBE_CSV, 2)


def mask_bad_bands(wl, y):
    y = y.copy()
    for lo, hi in BAD_BANDS:
        y[(wl >= lo) & (wl <= hi)] = np.nan
    return y


def load_importance(csv, target, smooth_nm=5):
    """Mean RF importance over folds on WL, lightly smoothed, scaled to its maximum; NaN in the water bands."""
    fi = pd.read_csv(csv)
    bcols = [c for c in fi.columns if c != "target_name"]
    wl = np.array([float(c) for c in bcols])
    imp = fi.loc[fi["target_name"] == target, bcols].to_numpy(dtype=float).mean(axis=0)
    imp = pd.Series(np.interp(WL, wl, imp)).rolling(smooth_nm, center=True, min_periods=1).mean().to_numpy()
    imp = mask_bad_bands(WL, imp)
    return imp / np.nanmax(imp)


def load_estimates(csv):
    df = pd.read_csv(csv)
    parts = df["full"].str.split("_", expand=True)
    df["block"], df["treat"] = parts[0], parts[1].map(TREAT_CODES)
    return df


def treatment_means(names):
    """{target: DataFrame indexed by treatment with meas, meas_sd, est, est_sd (%; block mean ± SD)}."""
    out = {}
    cache = {}
    for name in names:
        csv, col = TARGETS[name]["est"]
        if csv not in cache:
            cache[csv] = load_estimates(csv)
        df = cache[csv]
        plots = df.groupby(["block", "treat"])[[col, f"pred_{col}"]].mean() * 100
        g = plots.groupby("treat")
        out[name] = pd.DataFrame({"meas": g[col].mean(), "meas_sd": g[col].std(ddof=1),
                                  "est": g[f"pred_{col}"].mean(), "est_sd": g[f"pred_{col}"].std(ddof=1)}).loc[TREATS]
    return out


def r2_mae(meas, est):
    ss_res = np.sum((meas - est) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    return 1 - ss_res / ss_tot, np.mean(np.abs(est - meas))


# ================================================================ drawing helpers
def axes_in(fig, left, bottom, width, height, **kw):
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kw)


def heading(fig, x, y, letter, title):
    w, h = fig.get_size_inches()
    fig.text(x / w, y / h, letter, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="baseline")
    fig.text((x + 0.14) / w, y / h, title, fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="baseline")


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def water_bands(ax):
    for lo, hi in BAD_BANDS:
        ax.axvspan(lo, hi, color="#f0f0f0", lw=0, zorder=0)


def pigment_bands(ax):
    """Coloured strips of the pigment absorption regions (a.1 and a.2 share them)."""
    for _, (lo, hi), color in PIGMENTS:
        ax.axvspan(lo, hi, color=color, alpha=PIGMENT_ALPHA, lw=0, zorder=0)


def pigment_key(ax):
    """Colour key of the pigment strips, in a.1's empty upper-left area (right of the red-edge label)."""
    handles = [Patch(facecolor=color, alpha=PIGMENT_ALPHA, edgecolor="none") for _, _, color in PIGMENTS]
    key = ax.legend(handles=handles, labels=[p[0] for p in PIGMENTS], loc="upper left", bbox_to_anchor=(0.215, 0.95),
                    frameon=False, title="Pigment absorption", alignment="left", handlelength=1.2, handleheight=0.8,
                    handletextpad=0.5, labelspacing=0.3, borderaxespad=0, borderpad=0)
    key.get_title().set_fontweight("bold")
    ax.add_artist(key)            # a.1 also carries the biocrust-type legend


def feature_marks(ax, labels=True):
    """Vertical markers of the absorption features; rotated labels above the top axes (outside the data area)."""
    for label, pos in FEATURES:
        if isinstance(pos, tuple):
            ax.axvspan(pos[0], pos[1], color="#000000", alpha=0.05, lw=0, zorder=0)
            x = (pos[0] + pos[1]) / 2
        else:
            ax.axvline(pos, color="#777777", lw=0.5, ls=":", zorder=1)
            x = pos
        if labels:
            ax.text(x, 0.985, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5,
                    color="black", rotation=90, linespacing=0.95)
    if labels:
        for x in (1380, 1865):
            ax.text(x, 0.985, "Water\n(masked)", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=5, color="#555555", rotation=90, linespacing=0.95)


# ================================================================ panels
def panel_a1(ax):
    """Fingerprints: the four biocrust types, mean ± 1 SD, features annotated at the top."""
    spectra = load_probe()
    for name in BFT:
        t = TARGETS[name]
        refl = spectra[t["code"]]
        mean = mask_bad_bands(WL, refl.mean(axis=0))
        sd = mask_bad_bands(WL, refl.std(axis=0, ddof=1))
        ax.fill_between(WL, mean - sd, mean + sd, color=t["color"], alpha=0.15, lw=0, zorder=2)
        ax.plot(WL, mean, color=t["color"], lw=1.0, zorder=3, label=name)
        print(f"  a.1: {name:7s} n = {len(refl)} records")
    pigment_bands(ax)
    pigment_key(ax)
    water_bands(ax)
    feature_marks(ax, labels=True)
    ax.set_xlim(WL[0], WL[-1])
    ax.set_ylim(0, A1_YMAX)
    ax.set_yticks(np.arange(0, A1_YMAX, 0.2))
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Reflectance")
    ax.set_title("(a.1) Reflectance spectra", loc="left")
    clean(ax)
    # legend in the empty space below the curves, between the two water bands (n and ± 1 SD go in the caption)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.60, 0.0), handlelength=1.6, handletextpad=0.5,
              labelspacing=0.3)


def panel_mixed(ax, title):
    """Optional: subplot (mixed) spectra by treatment, mean ± 1 SD."""
    spectra = spectra_by_code(MIX_CSV, 1)
    for code, treat in TREAT_CODES.items():
        refl = spectra[code]
        mean = mask_bad_bands(WL, refl.mean(axis=0))
        sd = mask_bad_bands(WL, refl.std(axis=0, ddof=1))
        ax.fill_between(WL, mean - sd, mean + sd, color=TREAT_COLORS[treat], alpha=0.12, lw=0, zorder=2)
        ax.plot(WL, mean, color=TREAT_COLORS[treat], lw=1.0, zorder=3, label=f"{treat} (n = {len(refl)})")
    water_bands(ax)
    feature_marks(ax, labels=False)
    ax.set_xlim(WL[0], WL[-1])
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Reflectance")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.60, 0.0), handlelength=1.6, handletextpad=0.5,
              labelspacing=0.3)
    clean(ax)


def panel_a2(ax, title="(a.2) Random forest feature importance"):
    """Ridgeline of RF feature importance, one filled curve per target, scaled to its maximum."""
    n = len(RIDGES)
    step = 1.0
    for k, name in enumerate(RIDGES):
        t = TARGETS[name]
        imp = load_importance(*t["fi"])
        base = (n - 1 - k) * step
        ax.fill_between(WL, base, base + imp * 0.95 * step, where=~np.isnan(imp), color=t["color"], alpha=0.55,
                        lw=0, zorder=3 + k)
        ax.plot(WL, base + np.where(np.isnan(imp), np.nan, imp * 0.95 * step), color=t["color"], lw=0.5, zorder=3 + k)
        ax.axhline(base, color="#cccccc", lw=0.4, zorder=1)
        if name in RIDGE_SUB:   # composition of the stage, smaller, under the row label
            ax.text(-0.012, base + 0.27 * step, RIDGE_SUB[name], transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=5, color="#333333", linespacing=1.1)
        peak = WL[np.nanargmax(imp)]
        print(f"  a.2: {name:14s} importance peak at {peak:.0f} nm")
    pigment_bands(ax)
    water_bands(ax)
    feature_marks(ax, labels=False)
    ax.set_xlim(WL[0], WL[-1])
    ax.set_ylim(-0.05, n * step)
    ax.set_yticks([(n - 1 - k) * step + (0.64 if r in RIDGE_SUB else 0.3) for k, r in enumerate(RIDGES)])
    ax.set_yticklabels(RIDGES, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(title, loc="left")   # title instead of a y label: rows name themselves
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def panel_b(ax, name, letter, legend=False):
    """Estimated vs measured cover per subplot for one successional stage, points coloured by treatment."""
    csv, col = TARGETS[name]["est"]
    df = load_estimates(csv)
    meas, est = df[col].to_numpy() * 100, df[f"pred_{col}"].to_numpy() * 100
    r2, mae = r2_mae(meas, est)
    print(f"  b: {name:12s} n = {len(meas)}, R² = {r2:.3f}, MAE = {mae:.2f}%")
    for treat in TREATS:
        sel = (df["treat"] == treat).to_numpy()
        ax.scatter(meas[sel], est[sel], s=7, color=TREAT_COLORS[treat], alpha=0.85, edgecolor="white",
                   linewidth=0.3, zorder=3, label=treat)
    # OLS fit with its 95% confidence band for the mean (as seaborn.regplot draws it)
    slope, intercept = np.polyfit(meas, est, 1)
    xx = np.linspace(0, 90, 91)
    fit = slope * xx + intercept
    resid = est - (slope * meas + intercept)
    s = np.sqrt(np.sum(resid ** 2) / (len(meas) - 2))
    se = s * np.sqrt(1 / len(meas) + (xx - meas.mean()) ** 2 / np.sum((meas - meas.mean()) ** 2))
    try:
        from scipy import stats
        tval = stats.t.ppf(0.975, len(meas) - 2)
    except ImportError:
        tval = 1.97
    ax.fill_between(xx, fit - tval * se, fit + tval * se, color="#555555", alpha=0.2, lw=0, zorder=2)
    ax.plot(xx, fit, color="#333333", lw=0.8, zorder=4)
    ax.plot([0, 90], [0, 90], color="black", lw=0.5, ls="--", zorder=2)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.set_aspect("equal")
    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 30, 60, 90])
    ax.set_xlabel("Measured cover (%)")
    ax.set_ylabel("Estimated cover (%)")
    ax.set_title(f"({letter}) {name} biocrusts", loc="left")
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f}%\nn = {len(meas)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=5, linespacing=1.3)
    if legend:
        ax.legend(frameon=False, loc="lower right", markerscale=1.4, handletextpad=0.2, labelspacing=0.15,
                  title="Treatment", alignment="left")
    ax.tick_params(axis="x", pad=1)
    clean(ax)


def panel_c(ax, names, title, offsets, xlabels=True, kind_legend=True, ncol=2, stats_y=0.27, ymax=75,
            legend_loc="upper left"):
    """Treatment response curves: measured (solid, filled) and estimated (dashed, hollow), block mean ± SD."""
    stats = treatment_means(names)
    x = np.arange(len(TREATS))
    for name, off in zip(names, offsets):
        s, color = stats[name], TARGETS[name]["color"]
        ax.errorbar(x + off, s["meas"], yerr=s["meas_sd"], fmt="-o", color=color, lw=1.5, ms=3.8,
                    markeredgecolor=color, elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=4)
        ax.errorbar(x + off, s["est"], yerr=s["est_sd"], fmt="--o", color=color, lw=1.2, ms=3.8,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.9, elinewidth=0.6,
                    capsize=1.3, capthick=0.6, zorder=5)
        print(f"  c: {name:12s} measured " + ", ".join(f"{t} {v:.1f}" for t, v in s["meas"].items())
              + " | estimated " + ", ".join(f"{t} {v:.1f}" for t, v in s["est"].items()))
    # agreement of the plotted treatment means
    meas_all = np.concatenate([stats[n]["meas"].to_numpy() for n in names])
    est_all = np.concatenate([stats[n]["est"].to_numpy() for n in names])
    r2, mae = r2_mae(meas_all, est_all)
    print(f"  c: {title}: treatment means n = {len(meas_all)}, R² = {r2:.3f}, MAE = {mae:.2f}%")
    # statistics in the lower-right gap between the low-cover lines and the LtCy / early-successional error bars
    ax.text(0.97, stats_y, f"Treatment means (n = {len(meas_all)}):\n$R^2$ = {r2:.2f}, MAE = {mae:.1f}%",
            transform=ax.transAxes, ha="right", va="center", fontsize=5, color="#333333", linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(TREATS if xlabels else [])
    ax.set_xlim(-0.5, len(TREATS) - 0.5)
    ax.set_ylim(0, ymax)   # headroom for the two-row legends only
    ax.set_yticks(np.arange(0, ymax, 20 if ymax > 50 else 10))
    ax.set_ylabel("Fractional cover (%)")
    ax.set_title(title, loc="left")
    clean(ax)
    comp = [Line2D([], [], color=TARGETS[n]["color"], lw=1.5, marker="o", ms=3.8, label=n) for n in names]
    leg1 = ax.legend(handles=comp, frameon=False, loc=legend_loc, ncol=ncol, handlelength=1.6, columnspacing=0.8)
    ax.add_artist(leg1)
    if kind_legend:
        kind = [Line2D([], [], color="#444444", lw=1.5, marker="o", ms=3.8, label="Measured"),
                Line2D([], [], color="#444444", lw=1.2, ls="--", marker="o", ms=3.8, markerfacecolor="white",
                       markeredgewidth=0.9, label="Estimated from spectra")]
        ax.legend(handles=kind, frameon=False, loc="upper right", handlelength=2.2)


# ================================================================ figure
def main(show_mixed=False, out_name="Fig2"):
    """show_mixed: also draw the subplot spectra by treatment between the fingerprints and the importance."""
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(NATURE_RC)

    # one row, three columns: a (spectra over importance) | b.1/b.2 scatters | c.1/c.2 response curves
    LEFT, RIGHT = 0.58, 0.05                     # LEFT: two-line row labels of a.2
    HEAD, XLAB = 0.36, 0.34                      # HEAD holds the panel heading above the sub-panel titles
    a1_h, a2_h, a_gap = 1.25, 1.46, 0.26        # a.1 spectra, a.2 ridgeline (5 ridges); gap holds a.2's title
    am_h = 0.95 if show_mixed else 0.0           # optional mixed-spectra strip
    a_w = 2.78
    gap = 0.34                                   # column gap: just the y label and tick labels of the next column
    b_left = LEFT + a_w + gap
    b_s = 1.08                                   # b.1 / b.2 square size; the treatment legend sits between them
    c_left = b_left + b_s + gap
    c_w = FIG_W - RIGHT - c_left
    c_gap = 0.28                                 # room for the (c.2) title between the two stacked panels
    row_h = a1_h + a_gap + a2_h + (am_h + a_gap if show_mixed else 0)
    c_h = (row_h - c_gap) / 2
    H = HEAD + row_h + XLAB + 0.02
    fig = plt.figure(figsize=(FIG_W, H))

    top = H - HEAD
    heading(fig, 0.12, top + 0.20, "a", "Spectral fingerprints and model important wavelengths")
    panel_a1(axes_in(fig, LEFT, top - a1_h, a_w, a1_h))
    if show_mixed:
        panel_mixed(axes_in(fig, LEFT, top - a1_h - a_gap - am_h, a_w, am_h), "(a.2) Subplot spectra by treatment")
    panel_a2(axes_in(fig, LEFT, top - row_h, a_w, a2_h),
             "(a.3) Random forest feature importance" if show_mixed else "(a.2) Random forest feature importance")

    YLAB = 0.26                                  # panel letters b, c align with the left edge of the y labels
    heading(fig, b_left - YLAB, top + 0.20, "b", "Model accuracy")   # short: the column is 1.08 in wide
    panel_b(axes_in(fig, b_left, top - b_s, b_s, b_s), "Late succ.", "b.1")
    panel_b(axes_in(fig, b_left, top - row_h, b_s, b_s), "Early succ.", "b.2")
    # treatment colour key between b.1 and b.2 (below b.1's x label, above b.2's title)
    handles = [Line2D([], [], ls="none", marker="o", markersize=3.5, markerfacecolor=TREAT_COLORS[t],
                      markeredgecolor="white", markeredgewidth=0.3, label=t) for t in TREATS]
    band_top = top - b_s - 0.27                  # below b.1's x label
    band_bottom = top - row_h + b_s + 0.13       # above b.2's title
    key = fig.legend(handles=handles, loc="center", bbox_to_anchor=((b_left + b_s / 2) / FIG_W, (band_top + band_bottom) / 2 / H),
                     ncol=2, frameon=False, title="Treatment", alignment="left", handletextpad=0.1, columnspacing=0.6,
                     labelspacing=0.45, borderaxespad=0, borderpad=0)
    key.get_title().set_fontweight("bold")

    heading(fig, c_left - YLAB, top + 0.20, "c", "Response to the climate treatments")
    panel_c(axes_in(fig, c_left, top - c_h, c_w, c_h), STAGES, "(c.1) Successional stages",
            offsets=(-0.06, 0.06), xlabels=False, kind_legend=True, ncol=1, stats_y=0.26)
    ax_c2 = axes_in(fig, c_left, top - row_h, c_w, c_h)
    panel_c(ax_c2, BFT, "(c.2) Biocrust functional types", offsets=(-0.12, -0.04, 0.04, 0.12),
            xlabels=True, kind_legend=False, stats_y=0.20)
    ax_c2.set_xlabel("Treatment")

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{out_name}.{ext}")
        fig.savefig(out, dpi=1000, facecolor="white")
        print("wrote", out)
    plt.close(fig)
    print(f"figure {FIG_W / MM:.0f} x {H / MM:.0f} mm")


if __name__ == "__main__":
    main()   # main(show_mixed=True, out_name=...) adds the subplot spectra by treatment; not used in the paper
