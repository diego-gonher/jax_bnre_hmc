# Experiment Report

## Run Information

- **Experiment name**: {{exp_name}}
- **Run directory**: {{run_dir}}
- **Dataset file**: {{dataset_file}}
- **Model type**: {{model_type}}

---

## Training Summary

- **Status**: {{train_status}}
- **Best validation loss**: {{best_val_loss}}
- **Best epoch**: {{best_epoch}}

### Dimensions

- **Parameter dimension (theta_dim)**: {{theta_dim}}
- **Observation dimension (x_dim)**: {{x_dim}}

---

## Training Plots

### Loss curves

![Loss curves](../losses.pdf)

---

## HMC Summary

- **Status**: {{hmc_status}}

### Divergences

- **Total divergences**: {{divergences_total}}
- **Per observation**: {{divergences_per_observation}}
- **Per observation per chain**: {{divergences_per_observation_per_chain}}

### SBC Metrics

- **KS p-value (min)**: {{sbc_ks_pval_min}}
- **KS p-value (mean)**: {{sbc_ks_pval_mean}}

### TARP Metrics

- **MAE**: {{tarp_mae}}
- **IAE**: {{tarp_iae}}

---

## HMC Plots

### Corner plots (first three observations)

#### Observation 0
![Corner 0](corner_observation_0.pdf)

#### Observation 1
![Corner 1](corner_observation_1.pdf)

#### Observation 2
![Corner 2](corner_observation_2.pdf)

---

### TARP coverage curve

![TARP curve](tarp_ecp_curve.pdf)

---

### SBC rank histograms

![SBC ranks](sbc_rank_histograms.pdf)

---

## Posterior Samples

- **File**: {{posterior_samples_path}}

---

## Notes

- This report contains **raw outputs only**.
- No interpretation or conclusions are included.