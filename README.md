# MLB Pitcher Fatigue Detection

A pitch-by-pitch model that flags when a Major League Baseball starting
pitcher is getting tired, using only the public Statcast tracking data
that the league already collects.

This was my final project for DATA 534 (Applied Machine Learning) in the
M.S. Data Science and Analytics program at the College of Charleston.

If you've never trained a model before, that's fine. Follow the steps in
order and you'll end up with the same numbers reported in the paper.

---

## What this project does

Every pitch in MLB has its speed, spin, release point, and outcome
recorded by a system called **Statcast**. This project uses seven seasons
of that data (2019 through 2025, about 4.6 million pitches) to answer a
simple question:

> Can we tell, in real time, when a starting pitcher is getting fatigued,
> *before* the opposing team gains an advantage?

The short answer is yes. A gradient-boosted model trained on rolling
15-pitch windows raises a fatigue alert about 15 to 20 pitches before
opponent performance visibly improves, which is enough lead time for a
pitching coach to start warming up a reliever.

The full write-up is in [`final_report.pdf`](final_report.pdf).

---

## What you'll get when you run it

After running the pipeline you will have:

- A trained XGBoost model (`models/xgboost.pkl`) that takes a pitch as
  input and returns the probability that the pitcher is fatigued
- A summary CSV (`results/model_summary.csv`) with the test-set accuracy
  numbers for three models (logistic regression, random forest, XGBoost)
- A folder of plots (`plots/`) including model comparisons, feature
  importance, ROC and precision-recall curves, and a per-game fatigue
  trajectory

The headline number is **ROC-AUC = 0.881** for XGBoost on the held-out
2024 and 2025 seasons, which means the model is much better than chance
at distinguishing fatigued from non-fatigued pitches.

---

## Setup

### 1. Install Python

You need Python 3.10 or newer. If you don't have it:

- **Windows / Mac**: download from [python.org](https://www.python.org/downloads/)

To check what you have, open a terminal and run:

```bash
python --version
```

You should see `Python 3.10.x` or higher.

### 2. Clone this repository

```bash
git clone https://github.com/<your-username>/pitcher_fatigue.git
cd pitcher_fatigue
```

### 3. Create a virtual environment (recommended)

A virtual environment keeps this project's libraries separate from
anything else on your system.

```bash
python -m venv .venv
```

Activate it:

- **Mac / Linux**: `source .venv/bin/activate`
- **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
- **Windows (Git Bash)**: `source .venv/Scripts/activate`

You should see `(.venv)` at the start of your terminal prompt.

### 4. Install the libraries

```bash
pip install -r requirements.txt
```

This installs `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`,
`seaborn`, `pybaseball`, and `joblib`. Total install size is around 500 MB.

### 5. (Optional but recommended) Download the cached Statcast data

The pipeline downloads roughly 3 GB of pitch data from Baseball Savant
on its first run, which is the slowest stage. You can skip that download
by grabbing the cached parquet file from the Releases page:

1. Go to [https://github.com/braeden-mefford/pitcher_fatigue/releases](https://github.com/braeden-mefford/pitcher_fatigue/releases)
2. Download `statcast_2019_2025.parquet` from the latest release (~167 MB)
3. Place it at `data/statcast_2019_2025.parquet` inside your local copy of the repo

With the cached file in place, `python main.py` skips the download stage
and proceeds directly to feature engineering and model training. If you
skip this step, the pipeline will run end to end and download the data
itself.

> **Heads up on runtime.** Feature engineering iterates over 93,942
> pitcher-game appearances and is the most time-consuming stage. Total
> wall-clock time depends heavily on your hardware and disk speed,
> especially if the project lives on a synced cloud folder such as
> OneDrive, Dropbox, or Google Drive (sync overhead on every file write
> can slow things down considerably). Plan on the pipeline being a
> long-running job rather than something you wait on at your desk. Once
> you start it, leave it alone and check back later.

---

## Running the pipeline

The full pipeline has three stages:

1. **Download** the raw Statcast data from Baseball Savant
2. **Build features** (rolling statistics) and the fatigue label
3. **Train** the three models and save the results

You can run all three with one command:

```bash
python main.py
```

On the first run, the slowest stage is the data download,
which moves roughly 3 GB of pitch data from Baseball Savant in monthly
chunks. Subsequent runs re-use the cached download and skip straight to
feature engineering, which is then the slowest stage. See the runtime
note above. Treat this as a long-running job and let it run unattended.

### Running stages individually

If you only want to redo part of the pipeline, you can skip stages:

```bash
# skip the download (uses cached parquet)
python main.py --skip-download

# skip download AND feature engineering, train only
python main.py --skip-download --skip-features

# download only specific seasons
python main.py --years 2024 2025
```

### Looking at one pitcher's appearance

Once the model is trained, you can plot the in-game fatigue probability
for a specific start. You need the pitcher ID and the game ID:

```bash
python main.py --analyze --pitcher-id 592789 --game-pk 745455
```

This drops a PNG into `plots/`.

### Ranking pitchers by late-game fatigue in a season

```bash
python main.py --season-summary 2024
```

This scores every starting-pitcher appearance in 2024 and writes a CSV
ranking them by average fatigue probability after pitch 40.

---

## Project structure

```
pitcher_fatigue/
├── main.py                  # entry point, runs the full pipeline
├── data_collection.py       # downloads Statcast data via pybaseball
├── feature_engineering.py   # rolling-window features + fatigue label
├── modeling.py              # trains LR, RF, XGBoost; saves models + plots
├── create_visuals.py        # generates the four poster figures
├── requirements.txt         # Python dependencies
├── final_report.pdf         # the paper write-up (IEEE format)
├── data/
│   ├── raw/                       # per-season parquet files (created on first run)
│   ├── statcast_2019_2025.parquet # combined raw dataset
│   └── features.parquet           # feature dataset used for training
├── models/                  # saved trained models (.pkl)
├── plots/                   # all generated figures
└── results/
    └── model_summary.csv    # final ROC-AUC, PR-AUC, F1 numbers
```

---

## How the model works

### The data
Statcast records the speed, spin rate, release point, movement, plate
location, and batted-ball outcome of every pitch thrown in MLB. This
project pulls 4.6 million pitches from 2019-2025.

### The features
For every pitch, we look at the most recent 15 pitches that pitcher has
thrown in the current game and compute things like:

- Average fastball speed and how much it's dropping
- Spin rate and whether it's trending down
- How much the release point is wandering
- How hard hitters are squaring up the ball

We compare each pitch's stats to that pitcher's *own* stats from the
first 15 pitches of the game, which gives a per-pitcher, per-game baseline.

### The label
A pitch is labeled "fatigued" if both of these are true over the *next*
20 pitches:

1. Fastball speed drops by at least 1.5 mph below the early-game baseline
2. The rate of hard contact (95+ mph exit velocity) goes up by at least
   10 percentage points

The label looks **forward** in time, but the features only look
**backward**, so the model can never cheat by peeking at the answer.

### The split
We train on the 2019-2023 seasons, use 2023 to pick the decision
threshold, and only evaluate the final model on 2024 and 2025 data the
model has never seen.

### Three models
- **Logistic regression**: simple linear baseline
- **Random forest**: 300 decision trees voting together
- **XGBoost**: gradient-boosted trees, the best performer

---

## Glossary

| Term | What it means |
|---|---|
| Statcast | MLB's radar/optical tracking system, installed in every park |
| Pitch | One throw from the pitcher to the batter |
| Fastball | The pitcher's hardest, most-thrown pitch type |
| Offspeed | Slower pitches with more movement (curveball, slider, changeup) |
| Spin rate | How fast the ball is rotating, measured in rpm |
| Release point | The 3D coordinates where the pitcher lets go of the ball |
| Exit velocity | How fast the ball comes off the bat |
| Hard contact | A batted ball with exit velocity >= 95 mph |
| xwOBA | "Expected weighted on-base average" - a Statcast estimate of how good a batted ball was, regardless of whether it became a hit |
| ROC-AUC | A common accuracy metric. 0.5 = random guessing, 1.0 = perfect |

---

## Reproducing the exact paper results

If you run the pipeline end to end on the same Statcast data, you should
see the same numbers reported in the paper:

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.860 | 0.234 | 0.313 |
| Random Forest | 0.873 | 0.251 | 0.326 |
| **XGBoost** | **0.881** | **0.271** | **0.335** |

Note that Statcast occasionally retroactively adjusts past-season values
(usually small tweaks to xwOBA or barrel flags), so very minor differences
in the third decimal place are expected.

---

## What's original here vs. building on prior work

The closest published comparator is Kabra (2025), who used K-means
clustering on a single 2021 season to identify pitcher archetypes. This
project advances that work in three ways:

1. **Scale**: 7 seasons, 4.6M pitches vs. 1 season, ~700K pitches
2. **Features**: 38 engineered features across 6 categories
   (velocity, spin, release point, movement, command, contact quality)
   vs. just spin and velocity
3. **Output**: a per-pitch supervised probability with measurable lead time
   vs. a descriptive cluster summary

The full literature review and uniqueness analysis is in the paper.

---

## License and attribution

Code in this repository: MIT License (see `LICENSE`).

Data: MLB Statcast data is publicly available through Baseball Savant.
Access in this project is provided by the open-source
[pybaseball](https://github.com/jldbc/pybaseball) library by James LeDoux
and Moshe Schorr.

If you use this code or build on the methodology, please cite the paper
included in this repository.

---

## Contact

Braeden Mefford - bmefford19@gmail.com
M.S. Data Science and Analytics, College of Charleston
