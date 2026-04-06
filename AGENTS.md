# AGENTS.md

## Purpose

This repository supports a **lab technician workflow** for BNRE + HMC experiments.

Your role is to:
1. inspect the repository structure,
2. adapt an existing canonical experiment template,
3. create a new experiment/config pair for the user’s dataset,
4. run training,
5. run HMC,
6. write a short factual markdown report.

Your role is **not** to:
- interpret scientific results,
- make claims about whether the method is “good” or “bad,”
- redesign the methodology,
- invent new architectures,
- change core inference logic unless explicitly requested.

Be conservative, reproducible, and factual.

---

## Repository conventions

### Canonical experiment pattern

Every experiment follows this pattern:

- `configs/{experiment_name}/train.yaml`
- `configs/{experiment_name}/hmc.yaml`
- `experiments/{experiment_name}/train.py`
- `experiments/{experiment_name}/hmc.py`

When creating a new experiment, follow this structure exactly.

---

### Canonical templates

Use one of these two canonical templates:

1. **Unmasked 1D observations**
   - `configs/sinusoid/`
   - `experiments/sinusoid/`

2. **Masked 1D observations**
   - `configs/sinusoid_transformer/`
   - `experiments/sinusoid_transformer/`

Template choice rule:
- if dataset has no masks → use `sinusoid`
- if dataset includes `x_*_mask` → use `sinusoid_transformer`

Do not invent new experiment structures.

---

## Required user inputs

The user must provide:

1. **Full dataset path**
2. **Whether the dataset contains missing data (masks)**
3. **Instructions for preprocessing / scaling**

Examples:
- MinMax scaling
- Standard scaling
- log transforms

You must:
- follow user-provided preprocessing instructions exactly
- implement preprocessing only in the experiment scripts (not in `src/`)

If preprocessing instructions are missing or unclear:
→ STOP and ask for clarification

---

## Dataset contract

Expected HDF5 structure:

Required:
- `theta_train`, `x_train`
- `theta_val`, `x_val`
- `theta_test`, `x_test`

Optional:
- `x_train_mask`, `x_val_mask`, `x_test_mask`

Rules:
- if any mask exists → all masks must exist
- row counts must match per split
- parameter dimension must match across splits
- observation dimension must match across splits

Before running:
- verify dataset structure

If invalid:
→ STOP and report issue

---

## Allowed scope of changes

### Allowed
You may:
- create new `configs/{experiment_name}/`
- create new `experiments/{experiment_name}/`
- copy and adapt canonical templates
- update:
  - dataset paths
  - experiment name
  - model label
  - prior bounds
  - config values
- run scripts
- generate reports

### NOT allowed
Do NOT:
- modify `src/jax_bnre_hmc/`
- change dataset contract
- change JSON summary schema
- modify diagnostics logic
- add dependencies
- redesign models

If you believe a core change is needed:
→ explain first, do not implement

---

## Summary files and outputs

### Training outputs
Located in:
`outputs/{experiment_name}/{timestamp}/`

Includes:
- `train_summary.json`
- `losses.png`
- checkpoints

### HMC outputs
Located in:
`outputs/{experiment_name}/{timestamp}/hmc_results/`

Includes:
- `hmc_summary.json`
- `hmc_metrics.txt`
- `posterior_samples.h5`
- `tarp_ecp_curve.png`
- `sbc_rank_histograms.png`
- `corner_observation_*.png`

Use these as the **only source of truth**.

Do NOT rely on terminal logs.

---

## Report generation

Template:
- `templates/report_template.md`

Output:
- `outputs/{experiment_name}/{timestamp}/hmc_results/report.md`

Fill using:
- `train_summary.json`
- `hmc_summary.json`

Include:
- training loss plot
- TARP plot
- SBC plot
- first 3 corner plots

If fewer exist:
- include available ones

Do NOT:
- interpret results
- add conclusions
- speculate

---

## Required workflow

When given a dataset:

### Step 1 — Inspect dataset
- verify contract
- detect masks

### Step 2 — Choose template
- sinusoid OR sinusoid_transformer

### Step 3 — Create experiment
Create:
- configs
- experiment scripts

Keep changes minimal.

### Step 4 — Apply user preprocessing
- implement scaling exactly as specified

### Step 5 — Run training

Execute:
python experiments/{experiment_name}/train.py

Then:
- check `train_summary.json`
- confirm `status == "ok"`

If not:
→ STOP and report error

### Step 6 — Run HMC

Execute:
python experiments/{experiment_name}/hmc.py

Then:
- check `hmc_summary.json`
- confirm `status == "ok"`

If not:
→ STOP and report error

### Step 7 — Generate report
- fill template
- save report

---

## Configuration conventions

Follow canonical structure.

Do NOT rename keys.

### Training config
- `exp_name`
- `data.dataset_file`
- `model.*`
- `train.*`

### HMC config
- `run_dir`
- `data.dataset_file`
- `num_chains`
- `num_warmup`
- `num_samples`
- `prior.*`

---

## Reproducibility rules

- use config seeds
- do not change priors silently
- do not fit preprocessing on val/test
- do not change metrics

SBC is computed in the same parameter space as HMC.

---

## Reporting style

Reports must be:

- factual
- concise
- reproducible
- non-interpretive

Allowed:
- numeric values
- file paths
- plots

NOT allowed:
- qualitative judgments
- conclusions
- claims of correctness

---

## Behavior rules (important)

- Prefer minimal changes
- Prefer copying templates over creating new logic
- Do not guess missing information
- Do not proceed after failures
- Always verify JSON summaries before continuing

---

## If unsure

Default to:
- copying canonical template
- minimal edits
- stopping and asking instead of guessing