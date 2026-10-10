# HELIOS

**Intelligent energy management simulation for hybrid grid–solar–battery–generator systems, built around Uyo, Nigeria's unreliable power supply.**

HELIOS asks one question: *can a smarter decision layer, choosing between the public grid, solar panels, a battery and a backup generator, cut cost and fuel use compared to naive switching behavior?*

The project is a simulation and decision-logic study. It runs the same household, the same solar resource and the same outage history through several control strategies, then compares them on cost, generator runtime and unmet demand.

---

## Why this exists

In much of Uyo, grid power is intermittent. Households fall back on petrol or diesel generators, which are expensive, noisy and polluting. Adding solar and a battery helps, but only if the system decides *when* to charge, discharge, import from the grid or start the generator. A poor controller wastes the hardware. This project tests whether better rules, and a simple outage-risk predictor, make a measurable difference.

## How it works

### Data

| Input | Source |
|---|---|
| Solar irradiance | Real hourly data for Uyo from the [NASA POWER](https://power.larc.nasa.gov/) API (UTC shifted to local time) |
| Household demand | Synthetic hourly load curves, calibrated to a typical household |
| Grid availability | Synthetic outage model: outages are discrete events (start + duration), not independent hourly coin flips |

Outage scenarios: **mild, moderate, severe, extreme** (extreme targets roughly 70% grid availability). Every strategy sees the *same* hardware, solar, demand and outage history, so comparisons are fair.

### Strategies compared

- **Naive**: baseline switching with no foresight
- **Reactive**: responds to the current state only
- **HELIOS (static reserve)**: keeps a fixed battery reserve (30 / 50 / 70%) for outages
- **HELIOS (predictive)**: uses a logistic-regression outage-risk model to set the reserve dynamically
- **Oracle**: perfect-foresight benchmark, used as an upper bound on what any controller could achieve

### Metrics

Total cost, generator hours, fuel use and unserved energy, across four outage scenarios and multiple random seeds.

## Project structure

```
helios/
├── data/
│   ├── raw/solar/            # NASA POWER CSVs
│   ├── processed/            # solar_combined.csv, household_demand.csv,
│   │                         # grid_availability.csv, state_table.csv
│   └── results/              # experiment outputs
├── notebooks/
│   └── helios_experiment.ipynb
├── figures/
├── src/
│   ├── main.py
│   ├── data/                 # load_solar.py, generate_demand.py, outage generator
│   ├── simulation/           # config.py, solar.py, engine, strategies, risk.py
│   ├── run_experiment.py     # naive vs reactive vs HELIOS runs
│   ├── run_sweep.py          # multi-seed, evening-pattern and sizing sweeps
│   ├── train_risk.py         # trains the outage-risk predictor
│   └── tests/
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Build the input data

```bash
python src/data/load_solar.py
python src/data/generate_demand.py
```

This writes the solar, demand, grid availability and merged state table CSVs to `data/processed/`.

### Run the experiment

```bash
python src/run_experiment.py                     # all strategies, all outage scenarios
python src/run_sweep.py --pattern evening        # evening-heavy demand pattern
python src/run_sweep.py --sweep                  # PV / battery sizing sweep
python src/train_risk.py                         # train the outage-risk predictor
```

Results land in `data/results/`.

### Notebook

`notebooks/helios_experiment.ipynb` walks through the same pipeline step by step. It runs fine in Google Colab: upload the project zip, unzip it, and `%cd` into the folder.

### Tests

```bash
pytest src/tests
```

## Assumptions

All model assumptions live in `src/simulation/config.py`, split into three groups:

1. **Sourced**: taken from a reference (e.g. NASA POWER data)
2. **Estimated**: reasonable engineering estimates
3. **Scenario**: deliberately varied to test robustness

Default hardware: 5 kW PV array, 0.80 system efficiency, 5 kWh battery. Change these in `config.py` or via the sweep script.

## Results

> Fill this in with your final numbers before publishing.

| Scenario | Strategy | Cost | Generator hours | Unserved energy |
|---|---|---|---|---|
| Severe | Reactive | | | |
| Severe | HELIOS (predictive) | | | |
| Severe | Oracle | | | |

Key findings:

- _e.g. Static reserves cut generator hours but raise cost versus reactive._
- _e.g. Gains from smarter control are small in mild/moderate outages and largest in severe/extreme ones._
- _e.g. How close the predictive strategy gets to the oracle._

## Limitations

- Household demand and outage patterns are synthetic. They are calibrated, not measured.
- One year of solar data (2025) from a single location.
- Simplified battery model (no detailed degradation or temperature effects).
- Results depend on the assumptions in `config.py`.

## Roadmap

- Validate outage and load models against real metering data
- Add battery degradation and generator maintenance costs
- Results dashboard

## Data and acknowledgements

Solar data courtesy of the [NASA POWER Project](https://power.larc.nasa.gov/).

## Author

**Treasure Ani-Joseph**, Computer Science, University of Uyo
GitHub: [@treasure-cd](https://github.com/treasure-cd)
