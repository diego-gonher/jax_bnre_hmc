# ASTRA/Lightcone Compatibility POC

This directory is an optional proof-of-concept wrapper around the existing
sinusoid BNRE+HMC workflow. It does not modify training, HMC, diagnostics, or
report-generation code.

## What It Runs

The wrapper calls the existing entry points from the repository root:

```bash
python experiments/sinusoid/train.py
python experiments/sinusoid/hmc.py
python -m jax_bnre_hmc.report
```

If `datasets/sinusoid/sinusoid_noisy_no_masks.h5` is missing, the wrapper
generates it with:

```bash
python datasets/sinusoid/generate_dataset.py --out datasets/sinusoid/sinusoid_noisy_no_masks.h5
```

The dataset is documented as an ASTRA input, but `analysis_bundle` does not
declare it as a Lightcone rule input. This allows the wrapper to generate the
dataset if it is missing before the existing workflow starts.

## Profiles

- `smoke`: tiny non-scientific run for validating ASTRA/Lightcone wiring.
- `canonical`: existing sinusoid workflow scale using default train/HMC config
  values, with only `run_dir` passed explicitly to HMC.

## Proposed Validation Commands

From this directory:

```bash
astra validate astra.yaml
bash -n scripts/run_sinusoid_poc.sh
```

## Proposed Smoke Run

From this directory:

```bash
lc run --universe smoke analysis_bundle --jobs 1
lc status --universe smoke
lc verify --universe smoke
```

Canonical mode is documented but should not be run unless explicitly approved:

```bash
lc run --universe baseline analysis_bundle --jobs 1
```

## Outputs

Lightcone materializes:

```text
results/<universe>/analysis_bundle/
```

The wrapper copies key files from the Hydra run into that directory:

- `run_dir.txt`
- `train_summary.json`
- `hmc_summary.json`
- `hmc_metrics.txt` if present
- `report.md`
