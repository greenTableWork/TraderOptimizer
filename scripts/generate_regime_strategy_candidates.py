#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from trader_optimizer.data import find_trader_root
from trader_optimizer.strategy_candidate_generator import generate_strategy_candidate_pack


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        or Path.cwd()
        / "runs"
        / "strategy_candidate_universe"
        / f"plus{args.count}_non_cso_supported_{timestamp}"
    ).resolve()
    summary_output = (
        args.summary_output
        or output_dir.with_name(f"{output_dir.name}_summary.json")
    ).resolve()
    trader_root = (args.trader_root or find_trader_root()).resolve()
    summary = generate_strategy_candidate_pack(
        regime_vectors_path=args.regime_vectors,
        output_dir=output_dir,
        count=args.count,
        trader_root=trader_root,
        existing_config_globs=args.existing_config_glob,
        include_non_equity=args.include_non_equity,
        summary_output=summary_output,
    )
    if not args.quiet:
        print(f"candidate_count: {summary['candidateCount']}")
        print(f"symbols: {len(summary['symbols'])}")
        print(f"family_counts: {summary['familyCounts']}")
        print(
            "duplicate_existing_configs_skipped: "
            f"{summary['duplicateExistingConfigsSkipped']}"
        )
        print(f"output_dir: {summary['outputDir']}")
        print(f"summary: {summary_output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic non-CSO strategy candidates from regime-vector "
            "symbols for broad regime tuning universes."
        )
    )
    parser.add_argument(
        "--regime-vectors",
        type=Path,
        required=True,
        help="Regime vector JSONL produced by scripts/build_regime_vectors.py.",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Candidate config output directory. Defaults under runs/.",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--trader-root",
        type=Path,
        default=None,
        help="Trader workspace root. Defaults to auto-discovery from cwd.",
    )
    parser.add_argument(
        "--existing-config-glob",
        action="append",
        default=None,
        help=(
            "Existing strategy config glob relative to trader root. Used only for "
            "duplicate search-space suppression. May be repeated."
        ),
    )
    parser.add_argument(
        "--include-non-equity",
        action="store_true",
        help="Also generate STOCK-shaped configs for non-equity vector symbols.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
