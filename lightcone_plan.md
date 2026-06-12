• Markdown Plan

  ## 1. Workflow To Use

  Use the existing unmasked sinusoid workflow:

  - Training: experiments/sinusoid/train.py
  - HMC/diagnostics: experiments/sinusoid/hmc.py
  - Report: python -m jax_bnre_hmc.report

  Reasoning:

  - It is the simplest canonical workflow in AGENTS.md.
  - It uses the non-transformer MLP path.
  - It already follows the requested sequence: training -> HMC -> diagnostics -> markdown report.
  - It avoids masks, transformer-specific logic, and extra moving parts.

  Important note: datasets/sinusoid/sinusoid_noisy_no_masks.h5 is not currently present. The POC should either require the user to generate it first or include dataset generation as a
  preflight step using the existing generator.

  ## 2. Files To Add

  Keep everything isolated under lightcone/ so it is easy to delete:

  BNRE-HMC-lightcone/
    lightcone/
      README.md
      astra.yaml
      universes/
        baseline.yaml
        smoke.yaml
      scripts/
        run_sinusoid_poc.sh

  No changes to:

  - src/jax_bnre_hmc/
  - experiments/sinusoid/
  - configs/sinusoid/
  - diagnostics logic
  - training logic
  - HMC logic
  - project dependencies

  ## 3. What Each File Should Contain

  ### lightcone/astra.yaml

  Purpose: ASTRA/Lightcone analysis specification.

  Suggested structure:

  version: "0.0.10"
  name: "BNRE HMC Sinusoid Compatibility POC"
  description: |
    Proof-of-concept wrapper showing that the existing sinusoid BNRE+HMC
    workflow can be represented as an ASTRA/Lightcone-compatible reproducible
    analysis without changing the scientific code.

  inputs:
    - id: sinusoid_dataset
      type: data
      source: ../datasets/sinusoid/sinusoid_noisy_no_masks.h5
      description: "Unmasked sinusoid HDF5 benchmark dataset."

  outputs:
    - id: analysis_bundle
      type: data
      description: |
        Directory containing pointers/copies for the completed training run,
        HMC diagnostics, summaries, and markdown report.
      inputs: [sinusoid_dataset]
      decisions: [run_profile]
      recipe:
        command: bash scripts/run_sinusoid_poc.sh {output} {decisions.run_profile}

  decisions:
    run_profile:
      label: "Run Profile"
      rationale: "Smoke mode validates wiring quickly; canonical mode preserves default workflow scale."
      default: smoke
      options:
        smoke:
          label: "Smoke test"
          description: "Tiny run for plumbing validation only, not scientific."
        canonical:
          label: "Canonical workflow"
          description: "Uses the existing sinusoid workflow settings as closely as possible."

  Why one output instead of many:

  - The existing BNRE scripts write a Hydra run directory, not one Lightcone output directory per artifact.
  - A single analysis_bundle is the least invasive compatibility layer.
  - Lightcone can still manifest and verify the resulting output directory.

  ### lightcone/universes/baseline.yaml

  Purpose: canonical/default representation.

  decisions:
    run_profile: canonical

  ### lightcone/universes/smoke.yaml

  Purpose: quick validation without long experiments.

  decisions:
    run_profile: smoke

  ### lightcone/scripts/run_sinusoid_poc.sh

  Purpose: thin shell wrapper around existing commands.

  It should:

  1. Accept Lightcone output directory and profile.
  2. cd .. from lightcone/ to repo root.
  3. Check the sinusoid HDF5 exists.
  4. If missing, either fail with a clear message or generate it using the existing script.
  5. Run training with Hydra overrides.
  6. Detect the newest outputs/sinusoid/<timestamp> run directory created by training.
  7. Run HMC against that run directory.
  8. Generate markdown report.
  9. Write a small manifest-like pointer file into {output}.
  10. Optionally copy key summary files into {output} for Lightcone validation/readability.

  For smoke, use tiny overrides only for runtime validation:

  python experiments/sinusoid/train.py \
    train.epochs=2 \
    train.stop_after_epochs=null \
    train.save_every=1 \
    train.print_every=1 \
    data.dataset_file=datasets/sinusoid/sinusoid_noisy_no_masks.h5

  python experiments/sinusoid/hmc.py \
    run_dir="$RUN_DIR" \
    data.dataset_file=datasets/sinusoid/sinusoid_noisy_no_masks.h5 \
    num_chains=1 \
    num_warmup=2 \
    num_samples=2 \
    n_observations=1 \
    n_plots=1

  python -m jax_bnre_hmc.report --run-dir "$RUN_DIR" --num-corner-plots 1

  For canonical, use the existing workflow without scientific changes except explicit run_dir wiring:

  python experiments/sinusoid/train.py \
    data.dataset_file=datasets/sinusoid/sinusoid_noisy_no_masks.h5

  python experiments/sinusoid/hmc.py \
    run_dir="$RUN_DIR" \
    data.dataset_file=datasets/sinusoid/sinusoid_noisy_no_masks.h5

  python -m jax_bnre_hmc.report --run-dir "$RUN_DIR"

  Output directory contents should include something like:

  results/<universe>/analysis_bundle/
    run_dir.txt
    train_summary.json
    hmc_summary.json
    hmc_metrics.txt
    report.md

  Copying summaries/report into the Lightcone output directory is useful because Lightcone manifests hash that directory.

  ### lightcone/README.md

  Purpose: short human-facing usage note.

  Should document:

  - This is optional POC code.
  - It does not change BNRE/HMC logic.
  - How to validate ASTRA spec.
  - How to run smoke mode.
  - How to run canonical mode.
  - Where outputs appear.

  ## 4. Exact Commands The POC Would Run

  From repo root:

  ### Generate dataset if missing

  python datasets/sinusoid/generate_dataset.py \
    --out datasets/sinusoid/sinusoid_noisy_no_masks.h5

  ### Validate ASTRA spec

  cd lightcone
  astra validate astra.yaml

  ### Smoke execution through Lightcone

  cd lightcone
  lc run --universe smoke analysis_bundle --jobs 1
  lc status --universe smoke
  lc verify --universe smoke

  ### Canonical execution through Lightcone

  cd lightcone
  lc run --universe baseline analysis_bundle --jobs 1
  lc status --universe baseline
  lc verify --universe baseline

  ### Underlying smoke commands

  python experiments/sinusoid/train.py train.epochs=2 train.stop_after_epochs=null train.save_every=1 train.print_every=1 data.dataset_file=datasets/sinusoid/sinusoid_noisy_no_masks.h5

  python experiments/sinusoid/hmc.py run_dir="$RUN_DIR" data.dataset_file=datasets/sinusoid/sinusoid_noisy_no_masks.h5 num_chains=1 num_warmup=2 num_samples=2 n_observations=1 n_plots=1

  python -m jax_bnre_hmc.report --run-dir "$RUN_DIR" --num-corner-plots 1

  ## 5. How To Validate Without Running A Long Experiment

  Use only static and smoke checks:

  cd lightcone
  astra validate astra.yaml

  bash -n scripts/run_sinusoid_poc.sh

  lc run --universe smoke analysis_bundle --jobs 1

  Then check expected files:

  test -f results/smoke/analysis_bundle/train_summary.json
  test -f results/smoke/analysis_bundle/hmc_summary.json
  test -f results/smoke/analysis_bundle/report.md
  lc status --universe smoke
  lc verify --universe smoke

  The smoke run should be explicitly labeled non-scientific because it changes epochs, HMC samples, observations, and plots only to validate plumbing.

  ## 6. Risks / Unknowns

  - astra and lc are not project dependencies. That is acceptable, but docs must state they are optional external tools.
  - lc run expects outputs under results/<universe>/<output_id>; the wrapper must copy or summarize Hydra outputs into {output}.
  - The dataset file is absent right now. The wrapper should either fail clearly or generate it with the existing dataset generator.
  - Smoke mode may expose edge cases because HMC with num_samples=2 is numerically weak. It validates execution shape, not inference quality.
  - Canonical mode may be long: default training is 5000 epochs and HMC is 500 observations with 4 chains, 2000 warmup, and 2000 samples.
  - The existing configs/sinusoid/hmc.yaml has a hard-coded old run_dir; wrapper must always pass run_dir="$RUN_DIR" explicitly.
  - Lightcone container support should probably be omitted for this POC to avoid making BNRE dependencies part of a new container story.

  ## 7. Step-By-Step Implementation Plan

  1. Add isolated lightcone/ directory only.
  2. Add lightcone/astra.yaml with one analysis_bundle output and one run_profile decision.
  3. Add lightcone/universes/smoke.yaml and lightcone/universes/baseline.yaml.
  4. Add lightcone/scripts/run_sinusoid_poc.sh.
  5. Make the wrapper fail fast on missing tools, missing dataset, missing summaries, or failed statuses.
  6. Have the wrapper run existing scripts only, with Hydra CLI overrides.
  7. Have the wrapper copy train_summary.json, hmc_summary.json, hmc_metrics.txt, and report.md into {output}.
  8. Add lightcone/README.md explaining optional usage and smoke-vs-canonical semantics.
  9. Validate with astra validate astra.yaml.
  10. Validate shell syntax with bash -n.
  11. Run only smoke mode for end-to-end validation.
  12. Leave canonical mode documented but do not run it unless explicitly requested.
