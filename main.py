"""
End-to-end pipeline runner for the pitcher fatigue model.

Usage:
    python main.py                                  # full pipeline
    python main.py --skip-download --skip-features  # train only
    python main.py --analyze --pitcher-id 592789 --game-pk 745455
"""

import os
import argparse
import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_download(years, force=False):
    from data_collection import combine_seasons, SEASONS
    years = years or list(SEASONS.keys())
    log.info(f"Downloading seasons: {years}")
    return combine_seasons(years=years, force=force)


def run_feature_engineering(force=False):
    from feature_engineering import build_features
    log.info("Building features")
    return build_features(force=force)


def run_training():
    from modeling import train
    log.info("Training")
    return train()


def plot_appearance(pitcher_id: int, game_pk: int, model_name: str = "xgboost"):
    """Score a single pitcher-game and plot velocity, spin, and fatigue probability."""
    import joblib
    from feature_engineering import (
        add_game_pitch_number, compute_baseline, make_features,
    )
    from modeling import FEATURE_COLS as MODEL_FEATURES

    raw_path = os.path.join(BASE_DIR, "data", "statcast_2019_2025.parquet")
    model_path = os.path.join(BASE_DIR, "models", f"{model_name}.pkl")

    if not os.path.exists(raw_path):
        log.error(f"Raw data not found at {raw_path}.")
        return
    if not os.path.exists(model_path):
        log.error(f"Model not found at {model_path}.")
        return

    log.info("Loading raw data")
    df = pd.read_parquet(raw_path)

    appearance = df[(df["pitcher"] == pitcher_id) & (df["game_pk"] == game_pk)].copy()
    if appearance.empty:
        log.error(f"No pitches for pitcher {pitcher_id} in game {game_pk}.")
        return

    pitcher_name = appearance["player_name"].iloc[0] if "player_name" in appearance.columns else str(pitcher_id)
    game_date = appearance["game_date"].iloc[0]
    log.info(f"{pitcher_name}, game {game_pk} ({game_date}), {len(appearance)} pitches")

    appearance = add_game_pitch_number(appearance)
    appearance = compute_baseline(appearance)
    appearance = make_features(appearance)

    X = appearance[MODEL_FEATURES].astype(float)
    X = X.fillna(X.median())

    model = joblib.load(model_path)
    proba = model.predict_proba(X)[:, 1]
    appearance["fatigue_proba"] = proba

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"In-Game Fatigue Profile: {pitcher_name}\n"
        f"Game {game_pk}  |  {game_date}",
        fontsize=13, fontweight="bold"
    )

    pitch_nums = appearance["game_pitch_num"].values

    axes[0].plot(pitch_nums, appearance["release_speed"].values,
                 "o-", color="#2196F3", ms=4, lw=1.5, label="Release speed")
    if "baseline_velocity" in appearance.columns:
        axes[0].axhline(appearance["baseline_velocity"].iloc[0],
                        color="#2196F3", linestyle="--", lw=1,
                        label=f"Baseline ({appearance['baseline_velocity'].iloc[0]:.1f} mph)")
    axes[0].set_ylabel("Velocity (mph)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pitch_nums, appearance["release_spin_rate"].values,
                 "o-", color="#4CAF50", ms=4, lw=1.5, label="Spin rate")
    if "baseline_spin_rate" in appearance.columns:
        axes[1].axhline(appearance["baseline_spin_rate"].iloc[0],
                        color="#4CAF50", linestyle="--", lw=1,
                        label=f"Baseline ({appearance['baseline_spin_rate'].iloc[0]:.0f} rpm)")
    axes[1].set_ylabel("Spin rate (rpm)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(pitch_nums, proba, alpha=0.3, color="#FF5722")
    axes[2].plot(pitch_nums, proba, color="#FF5722", lw=2, label="Fatigue probability")
    axes[2].axhline(0.5, color="k", linestyle="--", lw=1, label="Decision threshold (0.50)")
    axes[2].set_ylabel("Fatigue probability")
    axes[2].set_xlabel("Pitch number in game")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fatigue_{pitcher_id}_{game_pk}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved {out_path}")

    print("\nPitch-level summary:")
    cols = ["game_pitch_num", "pitch_type", "release_speed", "release_spin_rate",
            "roll_velocity_mean", "velocity_decay", "roll_hc_rate", "fatigue_proba"]
    present = [c for c in cols if c in appearance.columns]
    print(appearance[present].to_string(index=False))


def season_fatigue_summary(year: int, model_name: str = "xgboost", top_n: int = 20):
    """Score every appearance in `year` and rank pitchers by late-game fatigue probability."""
    import joblib
    from feature_engineering import (
        add_game_pitch_number, compute_baseline, make_features,
    )
    from modeling import FEATURE_COLS as MODEL_FEATURES

    raw_path = os.path.join(BASE_DIR, "data", "statcast_2019_2025.parquet")
    model_path = os.path.join(BASE_DIR, "models", f"{model_name}.pkl")

    log.info(f"Loading raw data for {year}")
    df = pd.read_parquet(raw_path)
    df["year"] = pd.to_datetime(df["game_date"]).dt.year
    df = df[df["year"] == year].copy()

    model = joblib.load(model_path)
    log.info(f"Loaded {model_path}")

    df = add_game_pitch_number(df)
    df = compute_baseline(df)

    results = []
    groups = df.groupby(["game_pk", "pitcher"])
    from tqdm import tqdm
    for (game_pk, pitcher_id), g in tqdm(groups, desc="Scoring", unit="app"):
        if len(g) < 15:
            continue
        g_feat = make_features(g)
        X = g_feat[MODEL_FEATURES].astype(float).fillna(g_feat[MODEL_FEATURES].median())
        proba = model.predict_proba(X)[:, 1]
        late = proba[g_feat["game_pitch_num"].values >= 40]
        if len(late) == 0:
            continue
        results.append({
            "pitcher": pitcher_id,
            "player_name": g_feat["player_name"].iloc[0] if "player_name" in g_feat.columns else str(pitcher_id),
            "game_pk": game_pk,
            "game_date": g_feat["game_date"].iloc[0],
            "total_pitches": len(g_feat),
            "mean_late_fatigue": float(np.mean(late)),
            "max_fatigue_proba": float(np.max(proba)),
        })

    if not results:
        log.warning("No appearances with 40+ pitches found.")
        return pd.DataFrame()

    summary = pd.DataFrame(results).sort_values("mean_late_fatigue", ascending=False)
    out_path = os.path.join(BASE_DIR, "results", f"season_{year}_fatigue_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(f"\nTop {top_n} late-game fatigue appearances, {year}:")
    print(summary.head(top_n).to_string(index=False))
    log.info(f"Saved {out_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--pitcher-id", type=int)
    parser.add_argument("--game-pk", type=int)
    parser.add_argument("--model", default="xgboost",
                        choices=["logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--season-summary", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.analyze:
        if not args.pitcher_id or not args.game_pk:
            log.error("--analyze requires --pitcher-id and --game-pk.")
            return
        plot_appearance(args.pitcher_id, args.game_pk, model_name=args.model)
        return

    if args.season_summary:
        season_fatigue_summary(args.season_summary, model_name=args.model)
        return

    if not args.skip_download:
        import pybaseball as pb
        pb.cache.enable()
        run_download(args.years, force=args.force)
    else:
        log.info("Skipping download")

    if not args.skip_features:
        run_feature_engineering(force=args.force)
    else:
        log.info("Skipping features")

    if not args.skip_train:
        summary = run_training()
        print(summary.to_string(index=False))
    else:
        log.info("Skipping train")


if __name__ == "__main__":
    main()
