"""
Pulls Statcast pitch data from Baseball Savant via pybaseball
for the 2019-2025 regular seasons.
"""

import os
import time
import logging
from datetime import date, timedelta

import pandas as pd
import pybaseball as pb
from tqdm import tqdm


SEASONS = {
    2019: ("2019-03-20", "2019-09-30"),
    2020: ("2020-07-23", "2020-09-27"),   # COVID shortened
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

KEEP_COLS = [
    # ids
    "game_pk", "game_date", "pitcher", "player_name", "batter",
    "home_team", "away_team", "inning", "inning_topbot",
    "at_bat_number", "pitch_number",
    # pitch type
    "pitch_type", "pitch_name",
    # velocity / movement
    "release_speed", "effective_speed",
    "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z",
    # release point
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension",
    # plate location
    "plate_x", "plate_z", "zone",
    # outcome
    "description", "type", "events",
    # batted ball
    "launch_speed", "launch_angle", "hit_distance_sc",
    "bb_type", "barrel",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    # count
    "balls", "strikes", "outs_when_up",
    # score
    "home_score", "away_score",
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
COMBINED_PATH = os.path.join(os.path.dirname(__file__), "data", "statcast_2019_2025.parquet")
CHUNK_DAYS = 30

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def date_range_chunks(start: str, end: str, chunk_days: int = CHUNK_DAYS):
    cur = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_dt)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + timedelta(days=1)


def safe_statcast(start: str, end: str, retries: int = 3, wait: int = 10) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            return pb.statcast(start_dt=start, end_dt=end, verbose=False)
        except Exception as exc:
            log.warning(f"Attempt {attempt}/{retries} failed ({exc}). Retrying in {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"All {retries} attempts failed for {start} - {end}")


def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        log.warning(f"Columns not in API response: {missing}")
    return df[present].copy()


def download_season(year: int, force: bool = False) -> str:
    """Download all pitches for one season and save as parquet."""
    os.makedirs(RAW_DIR, exist_ok=True)
    season_path = os.path.join(RAW_DIR, f"statcast_{year}.parquet")

    if os.path.exists(season_path) and not force:
        log.info(f"{year}: cached, skipping.")
        return season_path

    start, end = SEASONS[year]
    chunks = list(date_range_chunks(start, end))
    frames = []

    log.info(f"Downloading {year} ({len(chunks)} chunks)")
    for chunk_start, chunk_end in tqdm(chunks, desc=f"{year}", unit="chunk"):
        df_chunk = safe_statcast(chunk_start, chunk_end)
        if df_chunk is not None and not df_chunk.empty:
            frames.append(trim_columns(df_chunk))
        time.sleep(1)

    if not frames:
        log.warning(f"No data returned for {year}.")
        return season_path

    season_df = pd.concat(frames, ignore_index=True)
    season_df["game_date"] = pd.to_datetime(season_df["game_date"])
    season_df["pitcher"] = season_df["pitcher"].astype("Int64")
    season_df["batter"] = season_df["batter"].astype("Int64")

    season_df.to_parquet(season_path, index=False)
    log.info(f"{year}: saved {len(season_df):,} pitches to {season_path}")
    return season_path


def combine_seasons(years: list = None, force: bool = False) -> pd.DataFrame:
    """Combine all season files into one parquet."""
    if years is None:
        years = list(SEASONS.keys())

    if os.path.exists(COMBINED_PATH) and not force:
        log.info(f"Combined dataset cached at {COMBINED_PATH}.")
        return pd.read_parquet(COMBINED_PATH)

    paths = [download_season(y) for y in years]
    frames = [pd.read_parquet(p) for p in paths if os.path.exists(p)]

    if not frames:
        raise RuntimeError("No season files found.")

    combined = pd.concat(frames, ignore_index=True)

    # dedupe on chunk-boundary duplicates
    id_cols = ["game_pk", "pitcher", "at_bat_number", "pitch_number"]
    combined = combined.drop_duplicates(subset=id_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(COMBINED_PATH), exist_ok=True)
    combined.to_parquet(COMBINED_PATH, index=False)
    log.info(f"Combined: {len(combined):,} pitches to {COMBINED_PATH}")
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(SEASONS.keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pb.cache.enable()

    df = combine_seasons(years=args.years, force=args.force)
    print(f"Total pitches: {len(df):,}")
