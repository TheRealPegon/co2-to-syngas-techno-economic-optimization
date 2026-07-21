"""
plot_sensitivity_tornado.py
---------------------------
Reads Plots/sensitivity_production_costs.csv and draws a tornado chart
showing the syngas production cost (€/t) for each parameter sweep, with
each bar pivoting on its flowsheet's base-case cost.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_HERE = Path(__file__).resolve().parent
_CSV  = _HERE.parent / "Plots" / "sensitivity_production_costs.csv"
_OUT  = _HERE.parent / "Plots" / "sensitivity_tornado.png"

COLORS = {"Membrane": "#4C72B0", "PSA": "#DD8452", "Amine": "#55A868"}
FS_ORDER = ["Membrane", "PSA", "Amine"]
FS_COLS  = {
    "Membrane": "Membrane_cost_eur_per_t",
    "PSA":      "PSA_cost_eur_per_t",
    "Amine":    "Amine_cost_eur_per_t",
}

# Display labels for the parameter rows (keyed on sweep_param), overriding the
# raw sweep_label stored in the CSV. Explicit "\n" sets the line breaks
# (used verbatim — the auto unit-split is skipped for overridden labels).
LABEL_OVERRIDES = {
    "price_power":    "Electricity Price\n[€/kWh]",
    "price_CO2_feed": "CO₂ Feed Price\n[€/t CO₂]",
    "base_cost_elec": "CO₂ Electrolyzer\nArea Cost [€/m²]",
}


def _param_range_labels(low_disp, high_disp, base_disp):
    if base_disp and base_disp != 0:
        lp = (low_disp  - base_disp) / abs(base_disp) * 100
        hp = (high_disp - base_disp) / abs(base_disp) * 100
        return (f"{lp:+.0f}%\n({low_disp:.4g})",
                f"{hp:+.0f}%\n({high_disp:.4g})")
    return f"{low_disp:.4g}", f"{high_disp:.4g}"


def main():
    df = pd.read_csv(_CSV)

    _PREFERRED_ORDER = ["price_power", "base_cost_elec", "price_CO2_feed"]
    all_params  = list(dict.fromkeys(df["sweep_param"]))
    param_order = sorted(
        all_params,
        key=lambda p: _PREFERRED_ORDER.index(p) if p in _PREFERRED_ORDER else len(_PREFERRED_ORDER),
    )

    records = []
    for param in param_order:
        grp  = df[df["sweep_param"] == param].reset_index(drop=True)
        lbl  = grp["sweep_label"].iloc[0]
        mid  = len(grp) // 2
        base = grp.iloc[mid]

        low_disp  = float(grp["display_value"].iloc[0])
        high_disp = float(grp["display_value"].iloc[-1])
        base_disp = float(base["display_value"])
        left_lbl, right_lbl = _param_range_labels(low_disp, high_disp, base_disp)

        per_fs = {}
        for fs in FS_ORDER:
            col       = FS_COLS[fs]
            base_c    = float(base[col])
            low_cost  = float(grp.iloc[0][col])
            high_cost = float(grp.iloc[-1][col])
            per_fs[fs] = (low_cost, high_cost, base_c)

        records.append({"param": param, "label": lbl,
                        "left_lbl": left_lbl, "right_lbl": right_lbl,
                        "per_fs": per_fs})

    # x-axis range based on absolute syngas production cost (€/t)
    all_costs = [c for r in records for fs in FS_ORDER
                 for c in r["per_fs"][fs] if not np.isnan(c)]
    c_min, c_max = min(all_costs), max(all_costs)
    span0   = max(c_max - c_min, 1.0)
    x_lo    = float(np.floor((c_min - span0 * 0.04) / 100) * 100)
    x_hi    = float(np.ceil((c_max + span0 * 0.04) / 100) * 100)
    span    = x_hi - x_lo

    # Grey box extends a little past the data range so the bars and their
    # flowsheet tip labels sit comfortably inside it, clear of the arrows.
    box_pad   = span * 0.12
    box_left  = x_lo - box_pad
    box_right = x_hi + box_pad
    text_x_l  = box_left - span * 0.07  # left arrow-label centre (fixed for all groups)
    text_x_r  = box_right + span * 0.10 # right arrow-label centre
    xlim = (text_x_l - span * 0.28, text_x_r + span * 0.12)

    # Layout
    BAR_H   = 0.26
    BAR_GAP = 0.06
    GRP_PAD = 0.14
    GRP_GAP = 0.44
    n_fs    = len(FS_ORDER)
    grp_h   = n_fs * BAR_H + (n_fs - 1) * BAR_GAP + 2 * GRP_PAD
    step    = grp_h + GRP_GAP

    fig_h = max(4.0, len(records) * step + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    arrowprops = dict(
        arrowstyle="->",
        color="#888888",
        lw=1.2,
        shrinkA=0,
        shrinkB=2,
    )

    for gi, rec in enumerate(records):
        y_bot = (len(records) - 1 - gi) * step
        y_top = y_bot + grp_h
        y_mid = (y_bot + y_top) / 2

        # Background box (data area only)
        ax.add_patch(FancyBboxPatch(
            xy=(box_left, y_bot), width=box_right - box_left, height=grp_h,
            boxstyle="round,pad=0.04", linewidth=0.7,
            edgecolor="#c0c0c0", facecolor="#f2f2f2", zorder=1, clip_on=False,
        ))

        # Arrows: tip pinned at box edges (fixed x), label outside at text_x_l/r
        # Left side: label left of box, arrow tip points right toward box
        ax.annotate(
            rec["left_lbl"],
            xy=(box_left, y_mid),           # arrowhead tip — always same x
            xytext=(text_x_l, y_mid),       # label position — always same x
            fontsize=8, ha="center", va="center",
            annotation_clip=False,
            arrowprops=arrowprops,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#d2d2d2",
                      edgecolor="#aaaaaa", linewidth=0.5),
        )
        # Right side: label right of box, arrow tip points left toward box
        ax.annotate(
            rec["right_lbl"],
            xy=(box_right, y_mid),
            xytext=(text_x_r, y_mid),
            fontsize=8, ha="center", va="center",
            annotation_clip=False,
            arrowprops=arrowprops,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#d2d2d2",
                      edgecolor="#aaaaaa", linewidth=0.5),
        )

        # Parameter label left of left arrow label
        if rec["param"] in LABEL_OVERRIDES:
            label_text = LABEL_OVERRIDES[rec["param"]]            # used verbatim
        else:
            label_text = rec["label"].replace(" [", "\n[")       # auto unit-split
        ax.text(text_x_l - span * 0.06, y_mid, label_text,
                va="center", ha="right", fontsize=9, clip_on=False, linespacing=1.3)

        # Bars: each pivots on its flowsheet's base cost, spanning toward the
        # low and high parameter values on the absolute €/t axis.
        for fi, fs in enumerate(FS_ORDER):
            low_cost, high_cost, base_c = rec["per_fs"][fs]
            y_bar = y_bot + GRP_PAD + fi * (BAR_H + BAR_GAP)
            color = COLORS[fs]

            ax.barh(y_bar, low_cost - base_c, height=BAR_H,
                    left=base_c, color=color, align="edge", zorder=3)
            ax.barh(y_bar, high_cost - base_c, height=BAR_H,
                    left=base_c, color=color, align="edge", zorder=3)

            # Base-case pivot marker for this bar
            ax.plot([base_c, base_c], [y_bar, y_bar + BAR_H],
                    color="black", linewidth=1.0, zorder=6, clip_on=False)

            # Flowsheet label at the outer (higher-cost) end of the span
            tip   = max(low_cost, high_cost)
            nudge = span * 0.015
            ax.text(tip + nudge, y_bar + BAR_H / 2, fs,
                    va="center", ha="left", fontsize=8,
                    color=color, fontweight="bold", zorder=5, clip_on=False)

    ax.set_xlim(xlim)
    ax.set_ylim(-GRP_GAP * 0.5,
                (len(records) - 1) * step + grp_h + GRP_GAP * 0.45)
    ax.set_yticks([])

    tick_step = 100.0 if span <= 800 else (250.0 if span <= 2000 else 500.0)
    ticks = np.arange(x_lo, x_hi + 1, tick_step)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Syngas Production Cost (€/t)", fontsize=10, labelpad=6)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 4))

    mid_frac = ((box_left + box_right) / 2 - xlim[0]) / (xlim[1] - xlim[0])
    ax.set_title("Sensitivity Analysis: Syngas Production Cost",
                 fontsize=14, fontweight="bold", pad=4, x=mid_frac, ha="center")

    fig.tight_layout()
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    print(f"Tornado plot saved: {_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
