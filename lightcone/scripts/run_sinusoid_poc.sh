#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 OUTPUT_DIR {smoke|canonical}" >&2
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTCONE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LIGHTCONE_DIR}/.." && pwd)"

OUTPUT_ARG="$1"
PROFILE="$2"

case "${PROFILE}" in
  smoke|canonical)
    ;;
  *)
    echo "Unknown run profile: ${PROFILE}" >&2
    usage
    exit 2
    ;;
esac

if [[ "${OUTPUT_ARG}" = /* ]]; then
  OUTPUT_DIR="${OUTPUT_ARG}"
else
  OUTPUT_DIR="${LIGHTCONE_DIR}/${OUTPUT_ARG}"
fi

DATASET="datasets/sinusoid/sinusoid_noisy_no_masks.h5"
DATASET_ABS="${REPO_ROOT}/${DATASET}"

cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${DATASET_ABS}" ]; then
  python datasets/sinusoid/generate_dataset.py --out "${DATASET}"
fi

before_runs="$(mktemp)"
after_runs="$(mktemp)"
trap 'rm -f "${before_runs}" "${after_runs}"' EXIT

find outputs/sinusoid -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort > "${before_runs}" || true

if [ "${PROFILE}" = "smoke" ]; then
  python experiments/sinusoid/train.py \
    train.epochs=2 \
    train.stop_after_epochs=null \
    train.save_every=1 \
    train.print_every=1 \
    data.dataset_file="${DATASET}"
else
  python experiments/sinusoid/train.py \
    data.dataset_file="${DATASET}"
fi

find outputs/sinusoid -mindepth 1 -maxdepth 1 -type d -print | sort > "${after_runs}"
RUN_DIR="$(comm -13 "${before_runs}" "${after_runs}" | tail -n 1)"

if [ -z "${RUN_DIR}" ]; then
  RUN_DIR="$(tail -n 1 "${after_runs}")"
fi

if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
  echo "Could not resolve the Hydra training run directory." >&2
  exit 1
fi

if [ "${PROFILE}" = "smoke" ]; then
  python experiments/sinusoid/hmc.py \
    run_dir="${RUN_DIR}" \
    data.dataset_file="${DATASET}" \
    num_chains=4 \
    num_warmup=100 \
    num_samples=100 \
    n_observations=10 \
    n_plots=1

  python -m jax_bnre_hmc.report \
    --run-dir "${RUN_DIR}" \
    --num-corner-plots 1
else
  python experiments/sinusoid/hmc.py \
    run_dir="${RUN_DIR}" \
    data.dataset_file="${DATASET}"

  python -m jax_bnre_hmc.report \
    --run-dir "${RUN_DIR}"
fi

HMC_DIR="${RUN_DIR}/hmc_results"

test -f "${RUN_DIR}/train_summary.json"
test -f "${HMC_DIR}/hmc_summary.json"
test -f "${HMC_DIR}/report.md"

cp "${RUN_DIR}/train_summary.json" "${OUTPUT_DIR}/train_summary.json"
cp "${HMC_DIR}/hmc_summary.json" "${OUTPUT_DIR}/hmc_summary.json"
cp "${HMC_DIR}/report.md" "${OUTPUT_DIR}/report.md"

if [ -f "${HMC_DIR}/hmc_metrics.txt" ]; then
  cp "${HMC_DIR}/hmc_metrics.txt" "${OUTPUT_DIR}/hmc_metrics.txt"
fi

{
  echo "profile: ${PROFILE}"
  echo "run_dir: ${RUN_DIR}"
  echo "dataset: ${DATASET}"
  echo "train_summary: ${RUN_DIR}/train_summary.json"
  echo "hmc_summary: ${HMC_DIR}/hmc_summary.json"
  echo "report: ${HMC_DIR}/report.md"
} > "${OUTPUT_DIR}/run_dir.txt"
