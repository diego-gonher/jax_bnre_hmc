from __future__ import annotations

import argparse
import json
from pathlib import Path

# Markers must match templates/report_template.md
_CORNER_SECTION_START = "### Corner plots (first three observations)\n"
_CORNER_TO_TARP = "\n---\n\n### TARP coverage curve\n"


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _maybe_corner_block(hmc_dir: Path, idx: int) -> str:
    corner_path = hmc_dir / f"corner_observation_{idx}.pdf"
    if not corner_path.is_file():
        return ""
    return (
        f"#### Observation {idx}\n"
        f"![Corner {idx}](corner_observation_{idx}.pdf)\n"
    )


def _dataset_file_from_train_yaml(train_yaml_text: str) -> str | None:
    for line in train_yaml_text.splitlines():
        if line.strip().startswith("dataset_file:"):
            return line.split(":", 1)[1].strip()
    return None


def _corner_heading(num_corner_plots: int) -> str:
    if num_corner_plots == 3:
        return _CORNER_SECTION_START
    return f"### Corner plots (first {num_corner_plots} observations)\n"


def generate_report(
    run_dir: Path,
    template_path: Path,
    output_path: Path | None = None,
    num_corner_plots: int = 3,
) -> Path:
    """Build ``report.md`` from the template and JSON/YAML under ``run_dir``."""
    run_dir = Path(run_dir).resolve()
    template_path = Path(template_path)
    hmc_dir = run_dir / "hmc_results"
    out = output_path if output_path is not None else hmc_dir / "report.md"
    out = Path(out).resolve()

    train_summary_path = run_dir / "train_summary.json"
    train_yaml_path = run_dir / "train.yaml"
    hmc_summary_path = hmc_dir / "hmc_summary.json"

    _require_file(train_summary_path, "train_summary.json")
    _require_file(train_yaml_path, "train.yaml")
    _require_file(hmc_summary_path, "hmc_summary.json")
    _require_file(template_path, "report template")

    train_summary = json.loads(train_summary_path.read_text())
    hmc_summary = json.loads(hmc_summary_path.read_text())
    train_cfg = train_yaml_path.read_text()
    template = template_path.read_text()

    exp_name = run_dir.parent.name
    dataset_file = _dataset_file_from_train_yaml(train_cfg)
    model_type = train_summary.get("model_type", "unknown")

    replacements = {
        "{{exp_name}}": exp_name,
        "{{run_dir}}": str(run_dir),
        "{{dataset_file}}": dataset_file or "unknown",
        "{{model_type}}": str(model_type),
        "{{train_status}}": str(train_summary.get("status", "unknown")),
        "{{best_val_loss}}": str(train_summary.get("best_val_loss", "n/a")),
        "{{best_epoch}}": str(train_summary.get("best_epoch", "n/a")),
        "{{theta_dim}}": str(train_summary.get("dims", {}).get("theta_dim", "n/a")),
        "{{x_dim}}": str(train_summary.get("dims", {}).get("x_dim", "n/a")),
        "{{hmc_status}}": str(hmc_summary.get("status", "unknown")),
        "{{divergences_total}}": str(hmc_summary.get("divergences_total", "n/a")),
        "{{divergences_per_observation}}": str(
            hmc_summary.get("divergences_per_observation", "n/a")
        ),
        "{{divergences_per_observation_per_chain}}": str(
            hmc_summary.get("divergences_per_observation_per_chain", "n/a")
        ),
        "{{sbc_ks_pval_min}}": str(hmc_summary.get("sbc_ks_pval_min", "n/a")),
        "{{sbc_ks_pval_mean}}": str(hmc_summary.get("sbc_ks_pval_mean", "n/a")),
        "{{tarp_mae}}": str(hmc_summary.get("tarp_mae", "n/a")),
        "{{tarp_iae}}": str(hmc_summary.get("tarp_iae", "n/a")),
        "{{posterior_samples_path}}": str(hmc_summary.get("posterior_samples_path", "n/a")),
    }

    report = template
    for key, value in replacements.items():
        report = report.replace(key, value)

    corner_sections = "\n".join(
        block
        for block in (
            _maybe_corner_block(hmc_dir, idx) for idx in range(int(num_corner_plots))
        )
        if block
    ).strip()

    if corner_sections:
        if _CORNER_SECTION_START not in report or _CORNER_TO_TARP not in report:
            raise ValueError(
                "Template is missing the expected corner-plot / TARP section markers; "
                "use templates/report_template.md or equivalent structure."
            )
        before, remainder = report.split(_CORNER_SECTION_START, 1)
        _, after = remainder.split(_CORNER_TO_TARP, 1)
        heading = _corner_heading(int(num_corner_plots))
        report = before + heading + "\n" + corner_sections + _CORNER_TO_TARP + after

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return out


def generate_report_for_experiment(
    exp_name: str,
    template_path: Path,
    run_dir: Path | None = None,
    num_corner_plots: int = 3,
) -> Path:
    """Resolve a run directory under ``outputs/<exp_name>/`` and generate a report."""
    if run_dir is None:
        base = Path("outputs") / exp_name
        if not base.is_dir():
            raise FileNotFoundError(f"No outputs directory for experiment: {base.resolve()}")
        candidates = sorted(p.resolve() for p in base.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No run directories found under {base.resolve()}")
        chosen = candidates[-1]
    else:
        chosen = Path(run_dir).resolve()

    return generate_report(
        chosen,
        template_path,
        output_path=None,
        num_corner_plots=num_corner_plots,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BNRE+HMC markdown report.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Hydra run directory (e.g. outputs/<exp_name>/<timestamp>).",
    )
    parser.add_argument(
        "--template-path",
        type=str,
        default="templates/report_template.md",
        help="Path to the markdown report template.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output path for the report; defaults to run_dir/hmc_results/report.md.",
    )
    parser.add_argument(
        "--num-corner-plots",
        type=int,
        default=3,
        help="Maximum number of corner plots to include (0 disables the section).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = generate_report(
        run_dir=Path(args.run_dir),
        template_path=Path(args.template_path),
        output_path=Path(args.output_path) if args.output_path is not None else None,
        num_corner_plots=int(args.num_corner_plots),
    )
    print(out)


if __name__ == "__main__":
    main()
