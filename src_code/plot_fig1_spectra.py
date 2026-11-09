"""Plot field reflectance spectra of the four biocrust types (mean ± 1 SD).

Input: 1_data/0_field_spectra/{LCY,DCY,Lichen,Moss}_Spectra.csv
    columns: wavelength (nm, 350-2500 at 1 nm), then one column per replicate spectrum.

Run with:
    /Users/jifujiang/miniconda3/envs/Fujiang/bin/python plot_biocrust_spectra.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/0_src_code/1_data/0_field_spectra"
FIG_DIR = "/Users/jifujiang/Desktop/2_University_of_Arizona/6_perspective_biocrusts/figures"

# file prefix -> (legend label, color); order = legend order (roughly successional)
CRUSTS = {
    "LCY":    ("Light cyanobacteria", "#2a78d6"),
    "DCY":    ("Dark cyanobacteria",  "#eb6834"),
    "Lichen": ("Lichen",              "#1baf7a"),
    "Moss":   ("Moss",                "#eda100"),
}

DIRECT_LABELS = True   # also label each mean line at its right end


def read_spectra(path):
    df = pd.read_csv(path)
    wl = df["wavelength"].to_numpy()
    refl = df.drop(columns="wavelength").to_numpy()    # (n_wavelength, n_replicate)
    return wl, refl


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

    ends = []   # (y at last wavelength, label, color) for direct labels
    for key, (label, color) in CRUSTS.items():
        wl, refl = read_spectra(os.path.join(DATA_DIR, f"{key}_Spectra.csv"))
        mean = refl.mean(axis=1)
        std = refl.std(axis=1, ddof=1)

        ax.fill_between(wl, mean - std, mean + std, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(wl, mean, color=color, lw=2, zorder=3, label=label)
        ends.append((mean[-1], label, color, wl[-1]))

    if DIRECT_LABELS:
        # nudge labels apart so close line ends do not overlap
        ends.sort(key=lambda e: e[0])
        min_gap = 0.028
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + min_gap)
        for y, (_, label, color, x) in zip(ys, ends):
            ax.plot([x, x + 25], [y, y], color=color, lw=2, clip_on=False)
            
    ax.set_xlim(350, 2500)
    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(500, 2501, 500))
    ax.set_xlabel("Wavelength (nm)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Reflectance", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8, color="#c3c2b7", labelcolor="#52514e")
    ax.grid(color="#e1e0d9", lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")

    ax.legend(loc="lower center", ncol=2, fontsize=10, frameon=False, handlelength=1.8,
              columnspacing=1.5, title="Mean ± 1 SD", title_fontsize=9)

    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"biocrust_spectra.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=1000)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
