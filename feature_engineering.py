"""
Build rolling-window fatigue features and the forward-looking
fatigue label from raw Statcast pitch data.
"""

import os
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


WINDOW = 15
BASELINE_N = 15
LABEL_WINDOW = 20

VEL_DROP_MPH = 1.5
HC_RATE_RISE = 0.10

HARD_CONTACT_MPH = 95.0

# 4-seam, sinker, cutter -> fastball; everything else is offspeed
FASTBALL_CODES = {"FF", "SI", "FC"}

FEATURE_PATH = os.path.join(os.path.dirname(__file__), "data", "features.parquet")
COMBINED_PATH = os.path.join(os.path.dirname(__file__), "data", "statcast_2019_2025.parquet")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def is_fastball(pitch_type):
    return pitch_type.isin(FASTBALL_CODES)


def is_hard_contact(launch_speed):
    return (launch_speed >= HARD_CONTACT_MPH).astype(float)


def is_barrel(launch_speed, launch_angle):
    # fallback definition when the Statcast `barrel` flag is missing
    flag = (launch_speed >= 98.0) & (launch_angle >= 26) & (launch_angle <= 30)
    return flag.fillna(False).astype(float)


def is_line_drive(bb_type):
    return (bb_type == "line_drive").astype(float)


def rolling_mean(s, w):
    return s.rolling(window=w, min_periods=1).mean()


def rolling_std(s, w):
    return s.rolling(window=w, min_periods=2).std()


def rolling_slope(s, w):
    """OLS slope of s over the last w observations."""
    def slope(arr):
        x = np.arange(len(arr))
        if len(arr) < 2 or np.all(np.isnan(arr)):
            return np.nan
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return np.nan
        return np.polyfit(x[valid], arr[valid], 1)[0]

    return s.rolling(window=w, min_periods=2).apply(slope, raw=True)


def add_game_pitch_number(df):
    df = df.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"])
    df["game_pitch_num"] = df.groupby(["game_pk", "pitcher"]).cumcount() + 1
    return df


def compute_baseline(df):
    """Per (pitcher, game) baseline values from the first 15 pitches."""
    early = df[df["game_pitch_num"] <= BASELINE_N].copy()
    early["is_fb"] = is_fastball(early["pitch_type"])

    vel_base = (
        early.groupby(["game_pk", "pitcher"])["release_speed"]
        .mean().rename("baseline_velocity")
    )

    early_fb = early[early["is_fb"]]
    fb_vel_base = (
        early_fb.groupby(["game_pk", "pitcher"])["release_speed"]
        .mean().rename("baseline_velocity_fb")
    )

    early_os = early[~early["is_fb"]]
    os_vel_base = (
        early_os.groupby(["game_pk", "pitcher"])["release_speed"]
        .mean().rename("baseline_velocity_os")
    )

    spin_base = (
        early.groupby(["game_pk", "pitcher"])["release_spin_rate"]
        .mean().rename("baseline_spin_rate")
    )

    early["is_hard_contact"] = is_hard_contact(early["launch_speed"])
    hc_base = (
        early.groupby(["game_pk", "pitcher"])["is_hard_contact"]
        .mean().rename("baseline_hc_rate")
    )

    early["xwoba_fill"] = early["estimated_woba_using_speedangle"].fillna(0.0)
    xwoba_base = (
        early.groupby(["game_pk", "pitcher"])["xwoba_fill"]
        .mean().rename("baseline_xwoba")
    )

    early["is_barrel"] = is_barrel(early["launch_speed"], early["launch_angle"])
    barrel_base = (
        early.groupby(["game_pk", "pitcher"])["is_barrel"]
        .mean().rename("baseline_barrel_rate")
    )

    df = (
        df.join(vel_base, on=["game_pk", "pitcher"])
          .join(fb_vel_base, on=["game_pk", "pitcher"])
          .join(os_vel_base, on=["game_pk", "pitcher"])
          .join(spin_base, on=["game_pk", "pitcher"])
          .join(hc_base, on=["game_pk", "pitcher"])
          .join(xwoba_base, on=["game_pk", "pitcher"])
          .join(barrel_base, on=["game_pk", "pitcher"])
    )
    return df


def make_features(group):
    """Compute rolling features for one (pitcher, game) appearance."""
    g = group.sort_values("game_pitch_num").copy()
    w = WINDOW

    g["is_fb"] = is_fastball(g["pitch_type"])

    g["roll_velocity_mean"]  = rolling_mean(g["release_speed"], w)
    g["roll_velocity_std"]   = rolling_std(g["release_speed"], w)
    g["roll_velocity_slope"] = rolling_slope(g["release_speed"], w)
    g["velocity_decay"]      = g["roll_velocity_mean"] - g["baseline_velocity"]
    g["velocity_drop_max"]   = (
        g["roll_velocity_mean"] - g["release_speed"]
    ).rolling(window=w, min_periods=1).max()

    # fastball-only velocity (NaN out offspeed so the rolling window only sees FBs)
    fb_vel = g["release_speed"].where(g["is_fb"])
    g["roll_fb_velocity_mean"]  = rolling_mean(fb_vel, w)
    g["roll_fb_velocity_std"]   = rolling_std(fb_vel, w)
    g["roll_fb_velocity_slope"] = rolling_slope(fb_vel, w)
    g["fb_velocity_decay"]      = g["roll_fb_velocity_mean"] - g["baseline_velocity_fb"]

    os_vel = g["release_speed"].where(~g["is_fb"])
    g["roll_os_velocity_mean"]  = rolling_mean(os_vel, w)
    g["roll_os_velocity_std"]   = rolling_std(os_vel, w)
    g["roll_os_velocity_slope"] = rolling_slope(os_vel, w)
    g["os_velocity_decay"]      = g["roll_os_velocity_mean"] - g["baseline_velocity_os"]

    g["roll_spin_mean"]  = rolling_mean(g["release_spin_rate"], w)
    g["roll_spin_std"]   = rolling_std(g["release_spin_rate"], w)
    g["roll_spin_slope"] = rolling_slope(g["release_spin_rate"], w)
    g["spin_decay"]      = g["roll_spin_mean"] - g["baseline_spin_rate"]

    g["roll_relx_std"] = rolling_std(g["release_pos_x"], w)
    g["roll_relz_std"] = rolling_std(g["release_pos_z"], w)
    g["roll_ext_mean"] = rolling_mean(g["release_extension"], w)
    g["roll_ext_std"]  = rolling_std(g["release_extension"], w)

    g["roll_pfxx_mean"] = rolling_mean(g["pfx_x"], w)
    g["roll_pfxz_mean"] = rolling_mean(g["pfx_z"], w)
    g["roll_pfxx_std"]  = rolling_std(g["pfx_x"], w)
    g["roll_pfxz_std"]  = rolling_std(g["pfx_z"], w)
    g["movement_mag"]   = np.sqrt(g["pfx_x"] ** 2 + g["pfx_z"] ** 2)
    g["roll_move_mean"] = rolling_mean(g["movement_mag"], w)

    g["roll_platex_std"] = rolling_std(g["plate_x"], w)
    g["roll_platez_std"] = rolling_std(g["plate_z"], w)
    in_zone = (
        (g["plate_x"].abs() <= 0.83) &
        (g["plate_z"] >= 1.5) & (g["plate_z"] <= 3.5)
    ).astype(float)
    g["roll_in_zone_rate"] = rolling_mean(in_zone, w)

    g["is_hard_contact"] = is_hard_contact(g["launch_speed"])
    g["roll_hc_rate"]    = rolling_mean(g["is_hard_contact"], w)
    g["hc_rate_rise"]    = g["roll_hc_rate"] - g["baseline_hc_rate"]

    g["xwoba_fill"]      = g["estimated_woba_using_speedangle"].fillna(0.0)
    g["roll_xwoba_mean"] = rolling_mean(g["xwoba_fill"], w)
    g["xwoba_rise"]      = g["roll_xwoba_mean"] - g["baseline_xwoba"]

    if "barrel" in g.columns and g["barrel"].notna().any():
        g["is_barrel"] = g["barrel"].fillna(0.0)
    else:
        g["is_barrel"] = is_barrel(g["launch_speed"], g["launch_angle"])
    g["roll_barrel_rate"] = rolling_mean(g["is_barrel"], w)

    g["is_line_drive"] = is_line_drive(g["bb_type"])
    g["roll_ld_rate"]  = rolling_mean(g["is_line_drive"], w)

    g["pitch_count"] = g["game_pitch_num"]
    g["is_strike"]   = (g["type"] == "S").astype(float)
    g["roll_strike_rate"] = rolling_mean(g["is_strike"], w)

    g["eff_speed_decay"] = (
        rolling_mean(g["effective_speed"], w)
        - rolling_mean(g["effective_speed"], w).iloc[:BASELINE_N].mean()
    )

    return g


def assign_fatigue_label(group):
    """
    Forward-looking label: pitch i is fatigued if the next LABEL_WINDOW
    pitches show both a fastball velocity drop and a hard-contact rise
    relative to that game's baseline.
    """
    g = group.sort_values("game_pitch_num").copy()
    n = len(g)
    labels = np.full(n, np.nan)

    vel      = g["release_speed"].to_numpy(dtype=float, na_value=np.nan)
    hc       = g["is_hard_contact"].to_numpy(dtype=float, na_value=np.nan)
    roll_hc  = g["roll_hc_rate"].to_numpy(dtype=float, na_value=np.nan)
    fb_mask  = g["is_fb"].to_numpy(dtype=bool)
    fb_base  = g["baseline_velocity_fb"].to_numpy(dtype=float, na_value=np.nan)

    for i in range(n - LABEL_WINDOW):
        future_slice = slice(i + 1, i + 1 + LABEL_WINDOW)
        future_hc = hc[future_slice]
        if len(future_hc) < LABEL_WINDOW:
            continue

        future_fb_mask = fb_mask[future_slice]
        future_fb_vel  = vel[future_slice][future_fb_mask]

        # need at least 3 fastballs for a stable mean
        if len(future_fb_vel) < 3 or np.all(np.isnan(future_fb_vel)):
            continue

        baseline_fb = fb_base[i]
        if np.isnan(baseline_fb):
            continue

        vel_cond = np.nanmean(future_fb_vel) <= (baseline_fb - VEL_DROP_MPH)

        current_hc = roll_hc[i]
        hc_cond = np.nanmean(future_hc) >= (current_hc + HC_RATE_RISE)

        labels[i] = int(vel_cond and hc_cond)

    g["fatigue_label"] = labels
    return g


# TODO: dedupe with FEATURE_COLS in modeling.py
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


def build_features(raw_path: str = COMBINED_PATH,
                   output_path: str = FEATURE_PATH,
                   force: bool = False) -> pd.DataFrame:
    """Run the full feature + label pipeline and cache to parquet."""
    if os.path.exists(output_path) and not force:
        log.info(f"Feature dataset cached at {output_path}.")
        return pd.read_parquet(output_path)

    log.info(f"Loading raw data from {raw_path}")
    df = pd.read_parquet(raw_path)
    log.info(f"Raw pitches: {len(df):,}")

    # filter out short outings (relievers, openers) -- need a stable baseline
    pitch_counts = df.groupby(["game_pk", "pitcher"]).size().rename("total_pitches")
    df = df.join(pitch_counts, on=["game_pk", "pitcher"])
    df = df[df["total_pitches"] >= WINDOW].drop(columns=["total_pitches"])
    log.info(f"After filtering short appearances: {len(df):,} pitches")

    df = add_game_pitch_number(df)
    df = compute_baseline(df)

    groups = df.groupby(["game_pk", "pitcher"])
    log.info(f"Building features for {len(groups):,} appearances")

    results = []
    BATCH_SIZE = 5_000
    batches = []
    for _, g in tqdm(groups, desc="features", unit="app"):
        g_feat = make_features(g)
        g_feat = assign_fatigue_label(g_feat)
        results.append(g_feat)
        if len(results) >= BATCH_SIZE:
            batches.append(pd.concat(results, ignore_index=True))
            results = []
    if results:
        batches.append(pd.concat(results, ignore_index=True))
        results = []
    featured = pd.concat(batches, ignore_index=True)
    del batches

    f64_cols = featured.select_dtypes(include=["float64"]).columns
    featured[f64_cols] = featured[f64_cols].astype("float32")

    featured = featured.dropna(subset=["fatigue_label"])
    log.info(f"Labelled pitches: {len(featured):,}")

    label_dist = featured["fatigue_label"].value_counts(normalize=True).round(4)
    log.info(f"Label distribution:\n{label_dist.to_string()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    featured.to_parquet(output_path, index=False)
    log.info(f"Saved features to {output_path}")
    return featured


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=COMBINED_PATH)
    parser.add_argument("--out", default=FEATURE_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = build_features(raw_path=args.raw, output_path=args.out, force=args.force)
    print(f"Feature dataset shape: {df.shape}")
    # print(df[FEATURE_COLS + ["fatigue_label"]].describe().to_string())
