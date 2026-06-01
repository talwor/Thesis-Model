# Bourke HIV & Influenza Co-infection Simulation

An agent-based, network-driven epidemiological model simulating the co-spread of HIV and influenza in a synthetic population based on Bourke, NSW (population ~2,340). Built as part of a final year thesis.

---

## Overview

The model represents each individual as a node in a dynamic sexual-contact network. Relationships form and dissolve over time, and disease spreads both through network edges and via community mixing. A key feature is the **co-infection interaction**: active influenza doubles the per-act HIV transmission probability, and HIV-positive individuals face elevated influenza susceptibility.

The simulation runs in daily timesteps over a configurable period (default 730 days / 2 years).

---



## Requirements

Python 3.10+ recommended (developed on 3.10).

```bash
pip install networkx matplotlib numpy
```

### Virtual environment setup (recommended on macOS if needed I had troubles on macOS so I just set up a virtual environment)

```bash
python3 -m venv venv
source venv/bin/activate
pip install networkx matplotlib numpy
```

---

## Running the Simulation

### Single run

```bash
python main.py
```

Runs a 730-day simulation on a population of 2,340 and displays prevalence/incidence plots on completion. Also outputs two `.graphml` files (start state and final state).

### Batch / control runs

```bash
# HIV only
python control_tests.py --mode hiv_only --days 730 --seeds 0-20

# Influenza only
python control_tests.py --mode flu_only --days 730 --seeds 0-20

# Both diseases, batch mode (saves CSVs)
python control_tests.py --mode both --days 730 --batch --seeds 0-20
```

`--seeds` accepts a range (`0-20`) or a single integer. Each seed sets `random.seed()` for reproducibility.

---

## Model Parameters

| Parameter | Value | Source |
|---|---|---|
| Population size | 2,340 | ABS Census, Bourke |
| Simulation length | 730 days | Thesis design |
| HIV seeds (initial infected) | 20 | Thesis design |
| Flu seeds (initial infected) | 20 | Thesis design |
| HIV transmission M→F (per act) | 1/1,234 | Literature |
| HIV transmission F→M (per act) | 1/2,380 | Literature |
| Flu co-infection HIV multiplier | ×2.0 | Literature |
| Flu edge transmission probability | 0.15 | Calibrated |
| Flu community contacts per step | 5 | Calibrated |
| Flu community beta | 0.10 | Calibrated |
| Flu incubation period | 4 days | Literature |
| Flu infectious period | 7 days | Literature |
| Flu immunity duration (mean) | 180 days | Literature |
| Relationship formation probability | 0.1429/day | ~1×/week |
| Breakup probability per step | 0.02 | Calibrated |
| Homophily (indigenous preference) | 0.70 | Calibrated |

---

## Disease Model

### HIV
- **States:** S (susceptible) → I (infected) → R (recovered after 180 days)
- Transmission occurs along network edges (sexual contacts)
- Probability doubles if either partner has active influenza

### Influenza
- **States:** S → E (exposed/incubating) → I (infectious) → R (recovered) → S (waned immunity)
- Spreads via network edges and community mixing
- HIV-positive individuals have 2× susceptibility
- Immunity wanes after ~180 days (heterogeneous, Gaussian distribution)

---

## Output

Each run produces:
- Console statistics (initialisation counts, final counts, relationships formed, breakups)
- `.graphml` files: the full network graph importable into NetworkX
- Matplotlib plots: HIV prevalence, HIV cumulative incidence, flu prevalence, flu daily incidence, flu cumulative incidence

Batch runs additionally export CSV files per metric (e.g. `batch_both_hiv_prevalence.csv`).

---

## Running Tests

```bash
python -m pytest tests/ -v
```

See `tests/test_simulation.py` for the full suite.
