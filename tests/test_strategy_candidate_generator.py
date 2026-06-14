import json
from pathlib import Path

from trader_optimizer.strategy_candidate_generator import (
    generate_strategy_candidate_pack,
    strategy_search_key,
)
from trader_optimizer.strategy_configs import discover_strategy_candidates


def _write_vector(path: Path, symbol: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"symbol": symbol}) + "\n")


def test_generate_strategy_candidate_pack_skips_non_equity_and_existing_duplicates(
    tmp_path: Path,
) -> None:
    vectors = tmp_path / "vectors.jsonl"
    _write_vector(vectors, "AAPL")
    _write_vector(vectors, "BTC")
    _write_vector(vectors, "MSFT")
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    existing_config = {
        "strategy_type": "MovingAverageCross",
        "fastWindow": 5,
        "slowWindow": 21,
        "contract": {"symbol": "AAPL"},
    }
    (existing_dir / "aapl_mac.json").write_text(
        json.dumps(existing_config),
        encoding="utf-8",
    )

    output_dir = tmp_path / "generated"
    summary = generate_strategy_candidate_pack(
        regime_vectors_path=vectors,
        output_dir=output_dir,
        count=5,
        trader_root=tmp_path,
        existing_config_globs=["existing/**/*.json"],
    )

    assert summary["candidateCount"] == 5
    assert summary["excludedSymbols"] == ["BTC"]
    assert summary["duplicateExistingConfigsSkipped"] == 1
    candidates = discover_strategy_candidates(tmp_path, ["generated/**/*.json"])
    assert len(candidates) == 5
    assert {symbol for candidate in candidates for symbol in candidate.symbols} <= {
        "AAPL",
        "MSFT",
    }
    generated_keys = {strategy_search_key(candidate.config) for candidate in candidates}
    assert strategy_search_key(existing_config) not in generated_keys


def test_generate_strategy_candidate_pack_reuses_existing_output_files(tmp_path: Path) -> None:
    vectors = tmp_path / "vectors.jsonl"
    _write_vector(vectors, "AAPL")
    _write_vector(vectors, "MSFT")
    output_dir = tmp_path / "generated"

    first = generate_strategy_candidate_pack(
        regime_vectors_path=vectors,
        output_dir=output_dir,
        count=4,
    )
    second = generate_strategy_candidate_pack(
        regime_vectors_path=vectors,
        output_dir=output_dir,
        count=4,
    )

    assert first["candidateCount"] == 4
    assert second["candidateCount"] == 4
    assert len(list(output_dir.glob("*.json"))) == 4
    assert all(
        record["reusedExistingPackFile"]
        for record in second["candidates"]
    )
