from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trader_optimizer.config import write_json
from trader_optimizer.regime_detector_specs import (
    DETECTOR_SPEC_SCHEMA,
    DETECTOR_SPEC_VERSION,
    detector_ids_for_vector,
    detector_spec_manifest,
    detector_specs_for_fields,
)
from trader_optimizer.regime_tuning_universe import (
    DEFAULT_REGIME_DIMENSIONS,
    regime_cell_for_vector,
    regime_cell_id,
)


LIVE_REGIME_DETECTION_SCHEMA = "live_regime_detection.v1"
LIVE_REGIME_STATE_FILE_SCHEMA = "live_regime_state_file.v1"
LIVE_REGIME_SUMMARY_SCHEMA = "live_regime_detection_summary.v1"


@dataclass(frozen=True)
class StrategyRegimeCandidate:
    symbol: str
    strategy_name: str
    config_path: str
    regime_cell_id: str
    regime_cell: Mapping[str, object]
    validation_status: str
    excess_return_pct: float | None = None
    spx_excess_return_pct: float | None = None
    same_stock_excess_return_pct: float | None = None
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "strategyName": self.strategy_name,
            "configPath": self.config_path,
            "regimeCellId": self.regime_cell_id,
            "regimeCell": dict(self.regime_cell),
            "validationStatus": self.validation_status,
            "excessReturnPct": self.excess_return_pct,
            "spxExcessReturnPct": self.spx_excess_return_pct,
            "sameStockExcessReturnPct": self.same_stock_excess_return_pct,
            "source": self.source,
        }


def build_live_regime_detection(
    vector: Mapping[str, object],
    *,
    previous_state: Mapping[str, object] | None = None,
    min_persistence: int = 3,
    change_point_threshold: float = 0.80,
    mode: str = "shadow",
    generated_utc: str | None = None,
    regime_dimensions: Sequence[str] = DEFAULT_REGIME_DIMENSIONS,
    strategy_candidates: Sequence[StrategyRegimeCandidate] = (),
) -> dict[str, object]:
    raw_cell = regime_cell_for_vector(vector, regime_dimensions)
    raw_cell_id = regime_cell_id(raw_cell)
    state = next_live_regime_state(
        vector,
        raw_cell=raw_cell,
        raw_cell_id=raw_cell_id,
        previous_state=previous_state,
        min_persistence=min_persistence,
        change_point_threshold=change_point_threshold,
    )
    selection = select_strategy_for_regime(
        symbol=str(vector.get("symbol") or ""),
        active_regime_cell_id=str(state["activeRegimeCellId"]),
        active_regime_cell=state["activeRegimeCell"],
        candidates=strategy_candidates,
    )
    core = vector.get("core")
    fields = set(raw_cell)
    if isinstance(core, Mapping):
        fields.update(str(key) for key in core)
    fields.update(_advanced_output_fields(vector))
    detection = {
        "schema": LIVE_REGIME_DETECTION_SCHEMA,
        "generatedUtc": generated_utc
        or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "mode": mode,
        "symbol": vector.get("symbol"),
        "barSize": vector.get("barSize"),
        "whatToShow": vector.get("whatToShow"),
        "useRth": vector.get("useRth"),
        "vectorStartUtc": vector.get("startUtc"),
        "vectorEndUtc": vector.get("endUtc"),
        "rawRegimeCellId": raw_cell_id,
        "rawRegimeCell": raw_cell,
        "activeRegimeCellId": state["activeRegimeCellId"],
        "activeRegimeCell": state["activeRegimeCell"],
        "transition": state["transition"],
        "state": state,
        "strategySelection": selection,
        "interWorlds": inter_world_contract(
            vector=vector,
            active_regime_cell_id=str(state["activeRegimeCellId"]),
            detector_fields=fields,
            selection=selection,
            mode=mode,
        ),
        "sourceVector": dict(vector),
    }
    return detection


def next_live_regime_state(
    vector: Mapping[str, object],
    *,
    raw_cell: Mapping[str, object],
    raw_cell_id: str,
    previous_state: Mapping[str, object] | None,
    min_persistence: int,
    change_point_threshold: float,
) -> dict[str, object]:
    symbol = str(vector.get("symbol") or "")
    vector_end = str(vector.get("endUtc") or "")
    change_point = _change_point_confidence(vector)
    if not previous_state:
        return {
            "schema": "live_regime_state.v1",
            "symbol": symbol,
            "stateKey": regime_state_key(vector),
            "activeRegimeCellId": raw_cell_id,
            "activeRegimeCell": dict(raw_cell),
            "activeSinceUtc": vector_end,
            "observationsInActiveCell": 1,
            "pendingRegimeCellId": None,
            "pendingRegimeCell": None,
            "pendingObservations": 0,
            "lastRawRegimeCellId": raw_cell_id,
            "lastVectorEndUtc": vector_end,
            "changePointConfidence": change_point,
            "stateConfidence": _state_confidence(vector, active_matches_raw=True),
            "transition": {
                "status": "initialized",
                "fromRegimeCellId": None,
                "toRegimeCellId": raw_cell_id,
                "reason": "first_observation",
            },
        }

    active_cell_id = str(previous_state.get("activeRegimeCellId") or "")
    if active_cell_id == raw_cell_id:
        return {
            **dict(previous_state),
            "schema": "live_regime_state.v1",
            "symbol": symbol,
            "stateKey": regime_state_key(vector),
            "activeRegimeCellId": active_cell_id,
            "activeRegimeCell": previous_state.get("activeRegimeCell") or dict(raw_cell),
            "observationsInActiveCell": int(
                previous_state.get("observationsInActiveCell") or 0
            )
            + 1,
            "pendingRegimeCellId": None,
            "pendingRegimeCell": None,
            "pendingObservations": 0,
            "lastRawRegimeCellId": raw_cell_id,
            "lastVectorEndUtc": vector_end,
            "changePointConfidence": change_point,
            "stateConfidence": _state_confidence(vector, active_matches_raw=True),
            "transition": {
                "status": "stable",
                "fromRegimeCellId": active_cell_id,
                "toRegimeCellId": active_cell_id,
                "reason": "raw_cell_matches_active",
            },
        }

    previous_pending_id = previous_state.get("pendingRegimeCellId")
    pending_observations = (
        int(previous_state.get("pendingObservations") or 0) + 1
        if previous_pending_id == raw_cell_id
        else 1
    )
    should_switch = (
        pending_observations >= max(1, min_persistence)
        or change_point >= change_point_threshold
    )
    if should_switch:
        reason = (
            "change_point_override"
            if change_point >= change_point_threshold
            else "hysteresis_confirmed"
        )
        return {
            "schema": "live_regime_state.v1",
            "symbol": symbol,
            "stateKey": regime_state_key(vector),
            "activeRegimeCellId": raw_cell_id,
            "activeRegimeCell": dict(raw_cell),
            "activeSinceUtc": vector_end,
            "observationsInActiveCell": 1,
            "pendingRegimeCellId": None,
            "pendingRegimeCell": None,
            "pendingObservations": 0,
            "lastRawRegimeCellId": raw_cell_id,
            "lastVectorEndUtc": vector_end,
            "changePointConfidence": change_point,
            "stateConfidence": _state_confidence(vector, active_matches_raw=True),
            "transition": {
                "status": "switched",
                "fromRegimeCellId": active_cell_id,
                "toRegimeCellId": raw_cell_id,
                "reason": reason,
                "pendingObservations": pending_observations,
            },
        }

    return {
        **dict(previous_state),
        "schema": "live_regime_state.v1",
        "symbol": symbol,
        "stateKey": regime_state_key(vector),
        "pendingRegimeCellId": raw_cell_id,
        "pendingRegimeCell": dict(raw_cell),
        "pendingObservations": pending_observations,
        "lastRawRegimeCellId": raw_cell_id,
        "lastVectorEndUtc": vector_end,
        "changePointConfidence": change_point,
        "stateConfidence": _state_confidence(vector, active_matches_raw=False),
        "transition": {
            "status": "held",
            "fromRegimeCellId": active_cell_id,
            "toRegimeCellId": active_cell_id,
            "rawRegimeCellId": raw_cell_id,
            "reason": "awaiting_hysteresis_confirmation",
            "pendingObservations": pending_observations,
            "requiredObservations": max(1, min_persistence),
        },
    }


def build_live_regime_detections(
    vectors: Sequence[Mapping[str, object]],
    *,
    previous_states: Mapping[str, Mapping[str, object]] | None = None,
    min_persistence: int = 3,
    change_point_threshold: float = 0.80,
    mode: str = "shadow",
    generated_utc: str | None = None,
    regime_dimensions: Sequence[str] = DEFAULT_REGIME_DIMENSIONS,
    strategy_candidates: Sequence[StrategyRegimeCandidate] = (),
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    states = dict(previous_states or {})
    detections: list[dict[str, object]] = []
    for vector in vectors:
        key = regime_state_key(vector)
        detection = build_live_regime_detection(
            vector,
            previous_state=states.get(key),
            min_persistence=min_persistence,
            change_point_threshold=change_point_threshold,
            mode=mode,
            generated_utc=generated_utc,
            regime_dimensions=regime_dimensions,
            strategy_candidates=strategy_candidates,
        )
        state = detection["state"]
        if isinstance(state, dict):
            states[key] = state
        detections.append(detection)
    return detections, states


def inter_world_contract(
    *,
    vector: Mapping[str, object],
    active_regime_cell_id: str,
    detector_fields: set[str],
    selection: Mapping[str, object],
    mode: str,
) -> dict[str, object]:
    return {
        "corpusWorld": {
            "detectorSpecSchema": DETECTOR_SPEC_SCHEMA,
            "detectorSpecVersion": DETECTOR_SPEC_VERSION,
            "detectors": detector_specs_for_fields(detector_fields),
            "rules": [
                "Every live detector must declare paper lineage or explicit heuristic status.",
                "Detector confidence is not a promotion benchmark.",
                "Online detectors must be causal and avoid full-sample leakage.",
            ],
        },
        "historicalOptimizerWorld": {
            "regimeVectorSchema": vector.get("schema"),
            "regimeCellId": active_regime_cell_id,
            "requiresBacktesterGatedConfig": True,
            "selectionStatus": selection.get("status"),
        },
        "liveRuntimeWorld": {
            "mode": mode,
            "allowedActions": ["observe", "select_validated_config"],
            "blockedActions": ["live_optimization", "ungated_config_promotion"],
        },
    }


def select_strategy_for_regime(
    *,
    symbol: str,
    active_regime_cell_id: str,
    active_regime_cell: object,
    candidates: Sequence[StrategyRegimeCandidate],
) -> dict[str, object]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.validation_status == "ok"
        and candidate.symbol.upper() in {symbol.upper(), "*"}
    ]
    if not eligible:
        return {
            "status": "no_validated_config",
            "reason": "No BackTester-gated strategy config is mapped to this symbol.",
            "selected": None,
        }

    exact = [
        candidate
        for candidate in eligible
        if candidate.regime_cell_id == active_regime_cell_id
    ]
    if exact:
        selected = _best_candidate(exact)
        return {
            "status": "exact_match",
            "reason": "Selected validated config with exact active regime cell.",
            "selected": selected.to_dict(),
        }

    active_cell = (
        active_regime_cell
        if isinstance(active_regime_cell, Mapping)
        else {}
    )
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            _regime_distance(active_cell, candidate.regime_cell),
            -_candidate_score(candidate),
        ),
    )
    selected = ranked[0]
    return {
        "status": "nearest_validated_config",
        "reason": "No exact regime match; selected nearest BackTester-gated regime cell.",
        "distance": _regime_distance(active_cell, selected.regime_cell),
        "selected": selected.to_dict(),
    }


def regime_state_key(vector: Mapping[str, object]) -> str:
    return "|".join(
        str(vector.get(field) if vector.get(field) is not None else "")
        for field in ("symbol", "barSize", "whatToShow", "useRth")
    )


def load_live_regime_state(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    states = payload.get("states") if isinstance(payload, dict) else None
    if not isinstance(states, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in states.items()
        if isinstance(value, Mapping)
    }


def write_live_regime_state(path: Path, states: Mapping[str, Mapping[str, object]]) -> None:
    write_json(
        path,
        {
            "schema": LIVE_REGIME_STATE_FILE_SCHEMA,
            "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "states": {key: dict(value) for key, value in sorted(states.items())},
        },
    )


def write_live_regime_detections_jsonl(
    path: Path,
    detections: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for detection in detections:
            handle.write(json.dumps(detection, sort_keys=True))
            handle.write("\n")


def write_live_regime_summary(
    path: Path,
    detections: Sequence[Mapping[str, object]],
) -> None:
    status_counts: dict[str, int] = {}
    selection_counts: dict[str, int] = {}
    by_symbol: dict[str, str] = {}
    for detection in detections:
        transition = detection.get("transition")
        if isinstance(transition, Mapping):
            status = str(transition.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        selection = detection.get("strategySelection")
        if isinstance(selection, Mapping):
            status = str(selection.get("status") or "unknown")
            selection_counts[status] = selection_counts.get(status, 0) + 1
        by_symbol[str(detection.get("symbol") or "")] = str(
            detection.get("activeRegimeCellId") or ""
        )
    write_json(
        path,
        {
            "schema": LIVE_REGIME_SUMMARY_SCHEMA,
            "generatedUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "detections": len(detections),
            "transitionStatusCounts": dict(sorted(status_counts.items())),
            "selectionStatusCounts": dict(sorted(selection_counts.items())),
            "activeRegimeCellBySymbol": dict(sorted(by_symbol.items())),
            "detectorSpecs": detector_spec_manifest(),
        },
    )


def load_strategy_regime_candidates(path: Path | None) -> list[StrategyRegimeCandidate]:
    if path is None:
        return []
    output: list[StrategyRegimeCandidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Strategy map line {line_number} must be a JSON object")
            output.append(strategy_regime_candidate_from_dict(payload))
    return output


def strategy_regime_candidate_from_dict(
    payload: Mapping[str, object],
) -> StrategyRegimeCandidate:
    cell = payload.get("regimeCell")
    if not isinstance(cell, Mapping):
        cell = {}
    return StrategyRegimeCandidate(
        symbol=str(payload.get("symbol") or "*").upper(),
        strategy_name=str(payload.get("strategyName") or payload.get("name") or ""),
        config_path=str(payload.get("configPath") or payload.get("bestConfig") or ""),
        regime_cell_id=str(payload.get("regimeCellId") or regime_cell_id(cell)),
        regime_cell=cell,
        validation_status=str(
            payload.get("validationStatus") or payload.get("status") or ""
        ),
        excess_return_pct=_float_or_none(payload.get("excessReturnPct")),
        spx_excess_return_pct=_float_or_none(payload.get("spxExcessReturnPct")),
        same_stock_excess_return_pct=_float_or_none(
            payload.get("sameStockExcessReturnPct")
        ),
        source=str(payload.get("source") or ""),
    )


def _change_point_confidence(vector: Mapping[str, object]) -> float:
    advanced = vector.get("advanced")
    if not isinstance(advanced, Mapping):
        return 0.0
    change_point = advanced.get("changePointConfidence")
    if isinstance(change_point, Mapping):
        return _bounded_float(change_point.get("confidence"))
    return _bounded_float(change_point)


def _state_confidence(
    vector: Mapping[str, object],
    *,
    active_matches_raw: bool,
) -> float:
    advanced = vector.get("advanced")
    persistence = 0.0
    if isinstance(advanced, Mapping):
        raw_persistence = advanced.get("regimePersistence")
        if isinstance(raw_persistence, Mapping):
            persistence = _bounded_float(raw_persistence.get("score"))
    if active_matches_raw:
        return max(0.5, persistence)
    return max(0.0, min(0.49, 1.0 - persistence))


def _advanced_output_fields(vector: Mapping[str, object]) -> set[str]:
    advanced = vector.get("advanced")
    if not isinstance(advanced, Mapping):
        return set()
    return {str(key) for key in advanced}


def _bounded_float(value: object) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return 0.0
    return min(1.0, max(0.0, numeric))


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _regime_distance(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> int:
    weights = {
        "directionSign": 4,
        "instrumentVolatilityRegime": 3,
        "marketVolatilityRegime": 2,
        "volumeRegime": 1,
    }
    return sum(
        weight
        for field, weight in weights.items()
        if str(left.get(field)) != str(right.get(field))
    )


def _best_candidate(
    candidates: Sequence[StrategyRegimeCandidate],
) -> StrategyRegimeCandidate:
    return sorted(candidates, key=lambda candidate: -_candidate_score(candidate))[0]


def _candidate_score(candidate: StrategyRegimeCandidate) -> float:
    scores = [
        value
        for value in (
            candidate.same_stock_excess_return_pct,
            candidate.spx_excess_return_pct,
            candidate.excess_return_pct,
        )
        if value is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0
