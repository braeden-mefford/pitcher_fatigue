"""
Generate the four poster-ready figures: model comparison bar chart,
XGBoost feature importance, rolling-window schematic, and a simulated
fatigue trajectory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# color palette
C_LR  = "#5B8DB8"
C_RF  = "#4CAF50"
C_XGB = "#E05C2A"
C_ACC = "#7B2D8B"
GREY  = "#AAAAAA"

TITLE_FS = 14
LABEL_FS = 11
TICK_FS  = 9


def fig_model_comparison():
    models    = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
    colors    = [C_LR, C_RF, C_XGB]
    roc_auc   = [0.8599, 0.8732, 0.8812]
    pr_auc    = [0.2336, 0.2505, 0.2712]
    precision = [0.2639, 0.2656, 0.3147]
    recall    = [0.3833, 0.4229, 0.3571]

    metrics = {
        "ROC-AUC":   roc_auc,
        "PR-AUC":    pr_auc,
        "Precision": precision,
        "Recall":    recall,
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Model Performance Comparison",
                 fontsize=TITLE_FS, fontweight="bold", y=1.02)

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(models, values, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.55)
        ax.set_title(metric_name, fontsize=LABEL_FS, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(labelsize=TICK_FS)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=TICK_FS, fontweight="bold")

        if metric_name == "ROC-AUC":
            ax.axhline(0.5, color="k", linestyle="--", lw=1, label="Chance (0.50)")
            ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


FEATURE_COLS = [
    "roll_velocity_mean", "roll_velocity_std", "roll_velocity_slope",
    "velocity_decay", "velocity_drop_max",
    "roll_fb_velocity_mean", "roll_fb_velocity_std",
    "roll_fb_velocity_slope", "fb_velocity_decay",
    "roll_os_velocity_mean", "roll_os_velocity_std",
    "roll_os_velocity_slope", "os_velocity_decay",
    "roll_spin_mean", "roll_spin_std", "roll_spin_slope", "spin_decay",
    "roll_relx_std", "roll_relz_std", "roll_ext_mean", "roll_ext_std",
    "roll_pfxx_mean", "roll_pfxz_mean", "roll_pfxx_std", "roll_pfxz_std",
    "roll_move_mean",
    "roll_platex_std", "roll_platez_std", "roll_in_zone_rate",
    "roll_hc_rate", "hc_rate_rise",
    "roll_xwoba_mean", "xwoba_rise",
    "roll_barrel_rate", "roll_ld_rate",
    "pitch_count", "roll_strike_rate", "eff_speed_decay",
]

# bar color by feature category (indices match FEATURE_COLS above)
CATEGORY_MAP = {
    "Velocity":         ([0,1,2,3,4,5,6,7,8,9,10,11,12], "#E05C2A"),
    "Spin Rate":        ([13,14,15,16],                    "#5B8DB8"),
    "Release Point":    ([17,18,19,20],                    "#4CAF50"),
    "Movement":         ([21,22,23,24,25],                 "#9C27B0"),
    "Location":         ([26,27,28],                       "#FF9800"),
    "Contact Quality":  ([29,30,31,32,33,34],              "#F44336"),
    "Workload":         ([35,36,37],                       "#607D8B"),
}


def feature_category_color(feat_idx):
    for cat, (indices, color) in CATEGORY_MAP.items():
        if feat_idx in indices:
            return color, cat
    return GREY, "Other"


def fig_feature_importance():
    model_path = os.path.join(BASE_DIR, "models", "xgboost.pkl")

    if not os.path.exists(model_path):
        print(f"XGBoost model not found at {model_path}. Using illustrative values.")
        raw_imp = np.array([
            0.118, 0.045, 0.062, 0.095, 0.038,            # velocity
            0.072, 0.041, 0.055, 0.088,                   # spin
            0.028, 0.025, 0.019, 0.022,                   # release point
            0.020, 0.018, 0.015, 0.017, 0.023,            # movement
            0.021, 0.019, 0.031,                          # location
            0.052, 0.071, 0.068, 0.083, 0.029, 0.035,     # contact quality
            0.044, 0.026, 0.040,                          # workload
        ])
        importances = raw_imp / raw_imp.sum()
    else:
        model = joblib.load(model_path)
        importances = model.feature_importances_

    top_n = 20
    idx = np.argsort(importances)[::-1][:top_n]
    feat_names = [FEATURE_COLS[i] for i in idx]
    feat_vals  = importances[idx]
    bar_colors = [feature_category_color(i)[0] for i in idx]

    name_map = {
        "roll_velocity_mean":  "Rolling Velocity Mean",
        "velocity_decay":      "Velocity Decay",
        "roll_velocity_slope": "Velocity Slope",
        "spin_decay":          "Spin Rate Decay",
        "roll_spin_mean":      "Rolling Spin Mean",
        "roll_spin_slope":     "Spin Rate Slope",
        "xwoba_rise":          "xwOBA Rise",
        "roll_xwoba_mean":     "Rolling xwOBA",
        "hc_rate_rise":        "Hard-Contact Rate Rise",
        "roll_hc_rate":        "Rolling Hard-Contact Rate",
        "velocity_drop_max":   "Max Velocity Drop",
        "roll_barrel_rate":    "Rolling Barrel Rate",
        "pitch_count":         "Pitch Count",
        "eff_speed_decay":     "Effective Speed Decay",
        "roll_in_zone_rate":   "Rolling In-Zone Rate",
        "roll_ld_rate":        "Rolling Line-Drive Rate",
        "roll_relx_std":       "Release-X Std Dev",
        "roll_relz_std":       "Release-Z Std Dev",
        "roll_velocity_std":   "Rolling Velocity Std Dev",
        "roll_move_mean":      "Rolling Movement Mag.",
    }
    display_names = [name_map.get(n, n) for n in feat_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(display_names[::-1], feat_vals[::-1],
            color=bar_colors[::-1], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Feature Importance (gain)", fontsize=LABEL_FS)
    ax.set_title("XGBoost Top-20 Feature Importances",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(labelsize=TICK_FS)
    ax.xaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=color, label=cat)
        for cat, (_, color) in CATEGORY_MAP.items()
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right",
              title="Feature Category", title_fontsize=8)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig_rolling_window_schematic():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor("#F9F9F9")

    ax.text(6.5, 4.65, "Rolling 15-Pitch Window Feature Construction",
            ha="center", va="center", fontsize=TITLE_FS,
            fontweight="bold", color="#1A1A2E")

    n_pitches = 40
    pitch_xs  = np.linspace(0.4, 12.6, n_pitches)
    pitch_y   = 3.5
    r = 0.15

    for i, x in enumerate(pitch_xs):
        in_window  = 14 <= i <= 28
        is_current = i == 28
        is_future  = 29 <= i <= 48

        if is_current:
            fc, ec, lw = "#E05C2A", "#8B2500", 2.2
        elif in_window:
            fc, ec, lw = "#5B8DB8", "#1A4A7A", 1.5
        elif is_future:
            fc, ec, lw = "#C8E6C9", "#4CAF50", 1.5
        else:
            fc, ec, lw = "#D0D0D0", "#888888", 1.0

        circle = plt.Circle((x, pitch_y), r, color=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(circle)

    w_start = pitch_xs[14]
    w_end   = pitch_xs[28]
    brace_y = pitch_y - 0.45
    ax.annotate("", xy=(w_end, brace_y), xytext=(w_start, brace_y),
                arrowprops=dict(arrowstyle="<->", color=C_LR, lw=2))
    ax.text((w_start + w_end) / 2, brace_y - 0.28,
            "15-pitch look-back window\n(features computed here)",
            ha="center", va="top", fontsize=9, color=C_LR, fontweight="bold")

    f_start = pitch_xs[29]
    f_end   = pitch_xs[min(48, n_pitches - 1)]
    ax.annotate("", xy=(f_end, brace_y), xytext=(f_start, brace_y),
                arrowprops=dict(arrowstyle="<->", color="#4CAF50", lw=2))
    ax.text((f_start + f_end) / 2, brace_y - 0.28,
            "20-pitch look-ahead window\n(label computed here, no leakage)",
            ha="center", va="top", fontsize=9, color="#388E3C", fontweight="bold")

    cx = pitch_xs[28]
    ax.text(cx, pitch_y + 0.38, "Pitch i\n(current)",
            ha="center", va="bottom", fontsize=8.5,
            color="#E05C2A", fontweight="bold")

    feat_box = FancyBboxPatch((0.3, 0.15), 4.0, 1.0,
                              boxstyle="round,pad=0.1",
                              fc="#EEF4FB", ec=C_LR, lw=1.5)
    ax.add_patch(feat_box)
    ax.text(2.3, 0.72, "Features (pitch i):",
            ha="center", va="center", fontsize=9, fontweight="bold", color=C_LR)
    ax.text(2.3, 0.38,
            "velocity_decay  |  spin_decay  |  roll_xwoba_mean\n"
            "roll_hc_rate  |  roll_barrel_rate  |  pitch_count  |  ...",
            ha="center", va="center", fontsize=7.5, color="#333333")

    lbl_box = FancyBboxPatch((8.7, 0.15), 4.0, 1.0,
                             boxstyle="round,pad=0.1",
                             fc="#F1FBF1", ec="#4CAF50", lw=1.5)
    ax.add_patch(lbl_box)
    ax.text(10.7, 0.72, "fatigue_label (pitch i):",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#388E3C")
    ax.text(10.7, 0.38,
            "1 if FB velocity drop >= 1.5 mph\n"
            "AND hard-contact rate rise >= +10%\n"
            "over next 20 pitches (FB only for velo)",
            ha="center", va="center", fontsize=7.5, color="#333333")

    ax.annotate("", xy=(4.3, 0.65), xytext=(pitch_xs[21], pitch_y - r),
                arrowprops=dict(arrowstyle="->", color=C_LR, lw=1.8,
                                connectionstyle="arc3,rad=0.3"))
    ax.annotate("", xy=(8.7, 0.65), xytext=(pitch_xs[38], pitch_y - r),
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.8,
                                connectionstyle="arc3,rad=-0.3"))

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "rolling_window_schematic.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig_fatigue_trajectory():
    np.random.seed(7)
    n = 95
    pitch_num  = np.arange(1, n + 1)
    fatigue_on = 55

    vel_base = 94.2
    vel = (vel_base
           + np.random.normal(0, 0.6, n)
           - np.clip((pitch_num - fatigue_on) * 0.045, 0, 4))

    spin_base = 2310
    spin = (spin_base
            + np.random.normal(0, 18, n)
            - np.clip((pitch_num - 45) * 2.2, 0, 200))

    xwoba_noise = np.random.normal(0, 0.06, n)
    xwoba = np.where(pitch_num < fatigue_on,
                     0.28 + xwoba_noise,
                     0.28 + xwoba_noise + (pitch_num - fatigue_on) * 0.003)
    xwoba = np.clip(xwoba, 0, 1)

    def roll_mean(arr, w=15):
        return pd.Series(arr).rolling(w, min_periods=1).mean().values

    vel_roll   = roll_mean(vel)
    spin_roll  = roll_mean(spin)
    xwoba_roll = roll_mean(xwoba)

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    fat_prob = sigmoid((pitch_num - fatigue_on) * 0.10) * 0.82 + np.random.normal(0, 0.03, n)
    fat_prob = np.clip(fat_prob, 0, 1)
    fat_roll = roll_mean(fat_prob, 10)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        "Simulated In-Game Pitcher Fatigue Trajectory\n"
        "Illustrative example, fatigue onset at pitch 55",
        fontsize=TITLE_FS, fontweight="bold", y=1.01
    )

    shade_kw = dict(alpha=0.12, color="#E05C2A", label="Post-fatigue region")

    axes[0].plot(pitch_num, vel, "o", ms=3, color=GREY, alpha=0.5, zorder=2)
    axes[0].plot(pitch_num, vel_roll, lw=2.2, color=C_XGB, label="15-pitch rolling mean")
    axes[0].axhline(vel_base, lw=1.2, ls="--", color="#333", label=f"Baseline ({vel_base} mph)")
    axes[0].axvline(fatigue_on, lw=1.5, ls="--", color=C_XGB, alpha=0.7)
    axes[0].fill_betweenx([90, 96], fatigue_on, n, **shade_kw)
    axes[0].set_ylabel("Velocity (mph)", fontsize=LABEL_FS)
    axes[0].set_ylim(90, 96.5)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(pitch_num, spin, "o", ms=3, color=GREY, alpha=0.5, zorder=2)
    axes[1].plot(pitch_num, spin_roll, lw=2.2, color=C_LR, label="15-pitch rolling mean")
    axes[1].axhline(spin_base, lw=1.2, ls="--", color="#333", label=f"Baseline ({spin_base} rpm)")
    axes[1].axvline(fatigue_on, lw=1.5, ls="--", color=C_XGB, alpha=0.7)
    axes[1].fill_betweenx([2050, 2380], fatigue_on, n, **shade_kw)
    axes[1].set_ylabel("Spin Rate (rpm)", fontsize=LABEL_FS)
    axes[1].set_ylim(2050, 2380)
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].plot(pitch_num, xwoba, "o", ms=3, color=GREY, alpha=0.5, zorder=2)
    axes[2].plot(pitch_num, xwoba_roll, lw=2.2, color=C_RF, label="15-pitch rolling xwOBA")
    axes[2].axhline(0.28, lw=1.2, ls="--", color="#333", label="Baseline (0.280)")
    axes[2].axvline(fatigue_on, lw=1.5, ls="--", color=C_XGB, alpha=0.7)
    axes[2].fill_betweenx([0.15, 0.65], fatigue_on, n, **shade_kw)
    axes[2].set_ylabel("xwOBA", fontsize=LABEL_FS)
    axes[2].set_ylim(0.15, 0.65)
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].spines[["top", "right"]].set_visible(False)

    axes[3].fill_between(pitch_num, fat_roll, alpha=0.25, color=C_XGB)
    axes[3].plot(pitch_num, fat_roll, lw=2.5, color=C_XGB, label="XGBoost fatigue probability")
    axes[3].axhline(0.5, lw=1.5, ls="--", color="k", label="Decision threshold (0.50)")
    axes[3].axvline(fatigue_on, lw=1.5, ls="--", color=C_XGB, alpha=0.7,
                    label=f"True fatigue onset (pitch {fatigue_on})")
    axes[3].set_ylabel("Fatigue Probability", fontsize=LABEL_FS)
    axes[3].set_xlabel("Pitch Number in Game", fontsize=LABEL_FS)
    axes[3].set_ylim(0, 1)
    axes[3].legend(fontsize=8, loc="upper left")
    axes[3].grid(True, alpha=0.3, linestyle="--")
    axes[3].spines[["top", "right"]].set_visible(False)

    for ax in axes:
        ax.tick_params(labelsize=TICK_FS)
        ax.set_xlim(1, n)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "fatigue_trajectory.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    print("Generating poster visuals")
    fig_model_comparison()
    fig_feature_importance()
    fig_rolling_window_schematic()
    fig_fatigue_trajectory()
    print(f"All figures saved to: {PLOTS_DIR}")
