# Formula 1 Race Strategy What-If Simulator

An interactive, explainable Formula 1 race strategy what-if simulator built as a Python web app. Starting from a validated real historical race (**2023 Singapore Grand Prix**, Round 15), users can modify strategy decisions, re-simulate the race lap by lap on a unified arithmetic engine, and compare simulated outcomes with what actually happened.

---

## Key Features

1. **Unified Lap-by-Lap Simulation Engine**:
   - Every what-if is an input control to a shared arithmetic engine, not an isolated feature.
   - Calculates cumulative times, relative gaps, pack bunching, and pit loss on every single lap ($1 \dots 62$).
   - Simulates alternate outcomes strictly under declared assumptions; never claims what definitely would have happened.

2. **3 Connected Machine Learning Models (100% Offline & Private)**:
   - **Model 1: Traffic & Overtaking Probability Classifier** (`LogisticRegression`, 98% accuracy):
     Evaluates pairwise proximity battles ($< 1.2\,\text{s}$) lap-by-lap based on pace delta, tyre age delta, and DRS. Trapped cars incur $+0.75\,\text{s/lap}$ dirty air turbulence penalties.
   - **Model 2: Non-Linear Tyre Degradation Cliff Regressor** (`HistGradientBoostingRegressor`, MAE 0.024s):
     Captures the 3 phases of tyre life (scrub-in, linear degradation, and thermal cliff breakdown at Soft L18, Med L28, Hard L42) and surfaces real-time stint health badges.
   - **Model 3: Local NLP Intent & Slot Extractor** (`TF-IDF + Calibrated Classifier`, 100% accuracy):
     Maps free-form strategy prompts into 10 intent classes and extracts driver, lap, and compound slots to drive the UI automatically.

3. **Multi-Page Architecture & Unified Cross-Linking**:
   - **🏁 Race Studio (`/`)**: Interactive strategy editor, scenario presets, composable interventions, side-by-side finishing order, and Plotly gap telemetry.
   - **🧠 Machine Learning Lab (`/ml-lab`)**: Interactive overtake calculator, Plotly tyre degradation curves, live NLP tester, driver battle presets, and one-click local retraining.
   - **📊 Validation Center (`/validation`)**: Replay Check reproducing official 10/10 order with penalty reconciliation, Train 60% / Test 40% held-out lap backtest, and data sanity checklists.

4. **Natural Language Strategy Assistant**:
   - Give prompts in plain English: *"make Sainz pit one lap earlier"*, *"what if there was no safety car?"*, *"switch Norris to a 2-stop strategy with Softs"*, *"optimize pit stop for Leclerc"*.
   - Ask analytical questions: *"Why did Verstappen win when the safety car was removed?"*, *"What is tyre degradation on Mediums vs Hards?"*, *"Who pitted under the lap 20 safety car?"*.

5. **Distinct Physical Track Conditions**:
   - **Green Flag**: $1.0\times$ pace multiplier, natural gap divergence.
   - **Virtual Safety Car (VSC)**: $1.35\times$ slowdown, **gaps strictly frozen** (no bunching), reduced pit loss ($\sim 13.0\,\text{s}$).
   - **Safety Car (SC)**: $1.45\times$ slowdown, **pack bunching active** (gaps compress at 70%/lap to $0.8\,\text{s}$ queue spacing), cheap pit stops ($\sim 11.5\,\text{s}$).

6. **Dual Validation Engine**:
   - **Replay Check**: Replays recorded laps through bookkeeping arithmetic to reproduce official finishing order ($10/10$ exact match) with explicit post-race penalty reconciliation (Sergio Perez 5.0s penalty).
   - **Model Backtest**: Evaluates model on held-out green laps (402 test laps) against a training-only stint median baseline. Reports scores honestly without artificial tuning.
   - **Sanity Checks**: Non-negative lap times, strictly monotonic cumulative times, and stint/pit count consistency.

---

## Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest Data & Train Offline Models (Optional, Pre-cached)
```bash
python -m src.milestone1_selector
python -m src.training.train_all
```

### 3. Launch Web Application
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

### 4. Run Automated Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Project Structure

```text
├── data/
│   ├── cache/
│   │   └── singapore_2023/       # Ingested & cached race datasets (offline ready)
│   └── models/                   # Pre-trained local ML models (.joblib)
├── src/
│   ├── app.py                    # FastAPI web server and multi-page endpoints
│   ├── milestone1_selector.py    # Automated race selector & data coverage check
│   ├── ai/
│   │   └── assistant.py          # Strategy AI Assistant & NLP copilot
│   ├── engine/
│   │   ├── simulation_engine.py  # Unified lap-by-lap arithmetic engine
│   │   └── scenario_manager.py   # Strategy & scenario builders
│   ├── ml/
│   │   ├── traffic_model.py      # Logistic regression overtake classifier
│   │   ├── tyre_cliff_model.py   # HistGradientBoosting non-linear tyre wear
│   │   └── nlp_intent_model.py   # TF-IDF + Logistic intent classifier
│   ├── models/
│   │   ├── pace_model.py         # Pace & degradation model
│   │   ├── pit_model.py          # Pit stop loss model & fallbacks
│   │   └── track_conditions.py   # SC bunching vs VSC frozen gap physics
│   ├── training/
│   │   └── train_all.py          # Local model training pipeline
│   └── validation/
│       ├── replay_check.py       # Replay check (reproduces 10/10 order)
│       ├── model_backtest.py     # Backtest on held-out laps vs baseline
│       └── sanity_checks.py      # Sanity & consistency checks
├── static/
│   ├── app.js                    # Multi-page frontend logic & Plotly charts
│   ├── style.css                 # Formula 1 dark UI theme styling
│   └── tutorial.js               # Interactive step-by-step guided tour
├── templates/
│   ├── base.html                 # Shared layout, navbar & styling
│   ├── index.html                # Race Studio dashboard
│   ├── ml_lab.html               # Machine Learning Lab
│   └── validation.html           # Validation & verification center
└── tests/
    ├── test_ai_assistant.py      # Assistant & NLP unit tests
    ├── test_api.py               # API & page integration tests
    ├── test_simulation_engine.py # Simulation engine unit tests
    └── test_validation.py        # Replay and backtest tests
```

---

## License

MIT License. Educational and analytical Formula 1 simulation project using public timing data from FastF1 and OpenF1.
