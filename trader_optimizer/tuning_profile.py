from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trader_optimizer.market_features import index_futures_for_symbol


TUNING_PROFILE_SCHEMA = "strategy_tuning_profile.v1"
DEFAULT_CURVE_SLOPE_SEVERITY = 3
ORDERBOOK_INTEGRATION_BRANCH = "codex/l2-orderbook-ingestion"

CRYPTO_SYMBOLS = {"BTC", "ETH", "LTC", "DOGE", "SOL", "XRP"}


def build_candidate_tuning_profile(
    candidate: Any,
    *,
    tuned_fields: Sequence[str],
    data_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    strategy_budget: float | None = None,
    market_features: Mapping[str, Any] | None = None,
    tuning_regions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    return build_tuning_profile(
        strategy_name=str(candidate.name),
        strategy_type=str(candidate.strategy_type),
        variant=str(candidate.variant),
        symbols=tuple(str(symbol) for symbol in candidate.symbols),
        config=candidate.config,
        tuned_fields=tuned_fields,
        data_profiles=data_profiles,
        hyperparameters=hyperparameters,
        strategy_budget=strategy_budget,
        market_features=market_features,
        tuning_regions=tuning_regions,
    )


def build_tuning_profile(
    *,
    strategy_name: str,
    strategy_type: str,
    variant: str,
    symbols: Sequence[str],
    config: Mapping[str, Any],
    tuned_fields: Sequence[str],
    data_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    strategy_budget: float | None = None,
    market_features: Mapping[str, Any] | None = None,
    tuning_regions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    normalized_symbols = tuple(str(symbol) for symbol in symbols)
    normalized_fields = tuple(str(field) for field in tuned_fields)
    contracts = _contracts_for_config(config, normalized_symbols)
    hyperparameter_names = sorted(str(key) for key in (hyperparameters or {}).keys())
    direction = _direction_category(strategy_type, variant, config, normalized_fields)
    volatility = _volatility_category(strategy_type, variant, normalized_fields)
    index_futures = _index_futures_category(contracts)
    options_probability = _options_probability_category(contracts)
    volume_orderbook = _volume_orderbook_category(data_profiles)
    _attach_market_feature_evidence(
        direction=direction,
        volatility=volatility,
        index_futures=index_futures,
        options_probability=options_probability,
        volume_orderbook=volume_orderbook,
        market_features=market_features,
    )

    profile: dict[str, object] = {
        "schema": TUNING_PROFILE_SCHEMA,
        "strategyName": strategy_name,
        "strategyType": strategy_type,
        "variant": variant,
        "symbols": list(normalized_symbols),
        "assetClasses": _asset_classes(contracts),
        "tunedFor": {
            "direction": direction,
            "volatility": volatility,
            "indexFuturesDirection": index_futures,
            "optionsProbabilityMap3d": options_probability,
            "tradeVolumeOrderbook": volume_orderbook,
        },
        "optimizedFor": {
            "primaryObjective": "maximize_train_validation_excess_return",
            "selectionScore": "0.70 train excess return + 0.30 validation excess return minus inventory, drawdown, and no-trade penalties",
            "backtesterGates": [
                "positive_strategy_return",
                "beat_spx_same_window",
                "beat_same_stock_buy_and_hold",
            ],
            "strategyBudget": strategy_budget,
            "hyperparameters": hyperparameter_names,
        },
    }
    normalized_tuning_regions = _normalize_tuning_regions(tuning_regions)
    if normalized_tuning_regions:
        profile["tuningRegions"] = normalized_tuning_regions
    profile["categoryLabels"] = tuning_category_labels(profile)
    return profile


def _attach_market_feature_evidence(
    *,
    direction: dict[str, object],
    volatility: dict[str, object],
    index_futures: dict[str, object],
    options_probability: dict[str, object],
    volume_orderbook: dict[str, object],
    market_features: Mapping[str, Any] | None,
) -> None:
    if not market_features:
        return
    direction_features = market_features.get("direction")
    if isinstance(direction_features, Mapping):
        direction["evidence"] = direction_features
        aggregate = direction_features.get("aggregate")
        if isinstance(aggregate, Mapping) and aggregate.get("curveSlopeSeverity") is not None:
            direction["observedCurveSlopeSeverity"] = aggregate["curveSlopeSeverity"]
            direction["observedDirection"] = aggregate.get("predictedDirection", "unknown")

    volatility_features = market_features.get("volatility")
    instrument_volatility = volatility.get("instrumentVolatility")
    if isinstance(volatility_features, Mapping) and isinstance(instrument_volatility, dict):
        instrument_volatility["evidence"] = volatility_features.get("bySymbol", {})
        market_by_future = volatility_features.get("marketByFuture")
        if market_by_future:
            market_volatility = volatility.get("marketVolatility")
            if isinstance(market_volatility, dict):
                market_volatility["evidence"] = market_by_future

    index_futures_features = market_features.get("indexFuturesDirection")
    if isinstance(index_futures_features, Mapping):
        index_futures["evidence"] = index_futures_features

    options_features = market_features.get("optionsProbabilityMap3d")
    if isinstance(options_features, Mapping) and options_features.get("status") == "active":
        options_probability["probabilityMap3dEvidence"] = options_features
        options_probability["status"] = "active"

    volume_features = market_features.get("volume")
    trade_volume = volume_orderbook.get("tradeVolume")
    if isinstance(volume_features, Mapping) and isinstance(trade_volume, dict):
        trade_volume["evidence"] = volume_features
        if volume_features.get("bySymbol"):
            trade_volume["status"] = "active"

    volume_orderbook_features = market_features.get("tradeVolumeOrderbook")
    orderbook = volume_orderbook.get("orderbook")
    if isinstance(volume_orderbook_features, Mapping):
        volume_orderbook["evidence"] = volume_orderbook_features
        if isinstance(orderbook, dict):
            orderbook["evidenceStatus"] = volume_orderbook_features.get(
                "orderbookStatus",
                "not_loaded",
            )


def tuning_category_labels(profile: Mapping[str, Any]) -> list[str]:
    tuned_for = profile.get("tunedFor")
    if not isinstance(tuned_for, Mapping):
        return []
    labels: list[str] = []
    for key, value in tuned_for.items():
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("status") or "")
        if status != "not_applicable":
            labels.append(str(key))
    return labels


def direction_plan_label(profile: Mapping[str, Any]) -> str:
    tuned_for = profile.get("tunedFor")
    if not isinstance(tuned_for, Mapping):
        return "unknown"
    direction = tuned_for.get("direction")
    if not isinstance(direction, Mapping):
        return "unknown"
    expected = direction.get("expectedDirection", "unknown")
    severity = direction.get("curveSlopeSeverity", DEFAULT_CURVE_SLOPE_SEVERITY)
    side_policy = direction.get("sidePolicy", "unknown")
    return f"{expected} / {side_policy} / slope {severity}"


def _direction_category(
    strategy_type: str,
    variant: str,
    config: Mapping[str, Any],
    tuned_fields: Sequence[str],
) -> dict[str, object]:
    expected_direction, side_policy, signal_source = _direction_bias(strategy_type, variant)
    return {
        "status": "active",
        "expectedDirection": expected_direction,
        "sidePolicy": side_policy,
        "curveSlopeSeverity": _curve_slope_severity(config),
        "curveSlopeSeverityDefault": DEFAULT_CURVE_SLOPE_SEVERITY,
        "curveSlopeSeverityMeaning": "higher means a steeper curve and stronger directional conviction",
        "signalSource": signal_source,
        "tunedFields": [
            field
            for field in tuned_fields
            if _field_matches(field, ("window", "baseline", "threshold", "step", "direction"))
        ],
    }


def _volatility_category(
    strategy_type: str,
    variant: str,
    tuned_fields: Sequence[str],
) -> dict[str, object]:
    instrument_fields = [
        field
        for field in tuned_fields
        if _field_matches(field, ("vol", "atr", "band", "z"))
    ]
    active = bool(instrument_fields) or (
        strategy_type == "PortfolioAllocation" and variant.upper() == "QS-001"
    )
    return {
        "status": "active" if active else "planned_input",
        "instrumentVolatility": {
            "status": "active" if active else "planned_input",
            "fields": instrument_fields,
            "purpose": "size exposure and classify whether the instrument itself is in a quiet or expanding regime",
        },
        "marketVolatility": {
            "status": "planned_input",
            "proxySymbols": ["SPX", "VIX", "ES"],
            "purpose": "separate broad-market volatility from idiosyncratic instrument volatility",
        },
    }


def _index_futures_category(contracts: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    applicable = []
    for contract in contracts:
        symbol = str(contract.get("symbol") or "").upper()
        sec_type = str(contract.get("secType") or contract.get("sec_type") or "").upper()
        if _is_crypto(sec_type, symbol):
            continue
        if sec_type in {"FUT", "FUTURE"}:
            applicable.append(
                {
                    "symbol": symbol,
                    "status": "direct_futures_instrument",
                    "candidateIndexFutures": [symbol],
                }
            )
            continue
        if sec_type in {"", "STK", "STOCK", "ETF", "IND"}:
            applicable.append(
                {
                    "symbol": symbol,
                    "status": "planned_input",
                    "candidateIndexFutures": index_futures_for_symbol(symbol),
                }
            )
    return {
        "status": "planned_input" if applicable else "not_applicable",
        "instruments": applicable,
        "insightMethod": (
            "compare index-future slope and session return against the strategy direction; "
            "confirm aligned moves and down-weight direction or momentum when futures conflict"
        ),
    }


def _options_probability_category(contracts: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    underlyings = [
        str(contract.get("symbol") or "").upper()
        for contract in contracts
        if not _is_crypto(
            str(contract.get("secType") or contract.get("sec_type") or "").upper(),
            str(contract.get("symbol") or "").upper(),
        )
    ]
    return {
        "status": "planned_input" if underlyings else "not_applicable",
        "underlyings": underlyings,
        "source": "options_trades",
        "probabilityMap3d": {
            "axes": ["expiration_days", "strike_moneyness", "trade_time_bucket"],
            "cellValues": [
                "up_probability",
                "down_probability",
                "momentum_probability",
                "call_put_premium_ratio",
                "trade_volume",
                "open_interest",
                "implied_volatility",
            ],
            "predictionTargets": ["direction", "momentum"],
        },
    }


def _volume_orderbook_category(
    data_profiles: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, object]:
    profiles = data_profiles or {}
    symbols_with_trade_bars = [
        symbol
        for symbol, profile in profiles.items()
        if str(profile.get("what_to_show") or "").upper() in {"TRADES", "AGGTRADES"}
    ]
    return {
        "status": "partial",
        "tradeVolume": {
            "status": "available_from_historical_bars" if symbols_with_trade_bars else "planned_input",
            "symbols": sorted(symbols_with_trade_bars),
            "features": ["bar_volume", "volume_z_score", "relative_volume"],
        },
        "orderbook": {
            "status": "external_branch_integration",
            "branch": ORDERBOOK_INTEGRATION_BRANCH,
            "strategyHook": "Strategy::onOrderBookUpdate",
            "features": ["bid_ask_imbalance", "book_pressure", "depth_slope"],
        },
        "predictionUse": "combine current volume impulse with orderbook imbalance to adjust direction and momentum confidence",
    }


def _contracts_for_config(
    config: Mapping[str, Any],
    symbols: Sequence[str],
) -> list[Mapping[str, Any]]:
    contracts: list[Mapping[str, Any]] = []
    raw_contracts = config.get("contracts")
    if isinstance(raw_contracts, Sequence) and not isinstance(raw_contracts, (str, bytes)):
        for contract in raw_contracts:
            if isinstance(contract, Mapping):
                nested_contract = contract.get("contract")
                contracts.append(nested_contract if isinstance(nested_contract, Mapping) else contract)
    for key in ("price_contract", "contract"):
        contract = config.get(key)
        if isinstance(contract, Mapping):
            contracts.append(contract)
    if contracts:
        return _dedupe_contracts(contracts)
    return [{"symbol": symbol} for symbol in symbols]


def _dedupe_contracts(contracts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Mapping[str, Any]] = []
    for contract in contracts:
        key = (
            str(contract.get("symbol") or "").upper(),
            str(contract.get("secType") or contract.get("sec_type") or "").upper(),
            str(contract.get("exchange") or "").upper(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(contract)
    return deduped


def _asset_classes(contracts: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            "symbol": str(contract.get("symbol") or "").upper(),
            "secType": str(contract.get("secType") or contract.get("sec_type") or "UNKNOWN").upper(),
            "exchange": str(contract.get("exchange") or "UNKNOWN"),
        }
        for contract in contracts
    ]


def _normalize_tuning_regions(
    tuning_regions: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, object]]:
    if not tuning_regions:
        return []
    fields = (
        "regionId",
        "role",
        "tuningSubcategory",
        "symbol",
        "period",
        "category",
        "barSize",
        "whatToShow",
        "useRth",
        "startUtc",
        "endUtc",
        "bucket",
        "bucketTimezone",
        "sourceManifest",
    )
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for region in tuning_regions:
        normalized = {
            field: region[field]
            for field in fields
            if field in region and region[field] is not None
        }
        if not normalized:
            continue
        region_id = str(normalized.get("regionId") or "")
        key = region_id or "|".join(
            str(normalized.get(field, ""))
            for field in (
                "role",
                "tuningSubcategory",
                "symbol",
                "period",
                "category",
                "startUtc",
                "endUtc",
            )
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def _direction_bias(strategy_type: str, variant: str) -> tuple[str, str, str]:
    normalized_variant = variant.upper()
    if strategy_type == "ConstantStepOffset":
        return "up_or_down_around_baseline", "mean_reversion_band", "price_curve"
    if strategy_type == "MovingAverageCross":
        return "up", "long_flat", "moving_average_slope"
    if strategy_type == "TechnicalSignal":
        if normalized_variant in {"TS-005", "RSIDIVERGENCE", "RSI_DIVERGENCE"}:
            return "up_or_down", "long_short_reversal", "price_rsi_divergence"
        return "up", "long_flat_breakout", "price_curve"
    if strategy_type == "PortfolioAllocation":
        if normalized_variant == "QS-001":
            return "up_scaled_by_volatility", "adaptive_long_exposure", "basket_volatility"
        if normalized_variant == "QS-002":
            return "up_or_down", "cross_sectional_long_short", "relative_momentum"
        if normalized_variant == "PAIRS-001":
            return "relative_up_or_down", "market_neutral_pair_spread", "pair_spread"
    return "unknown", "unknown", "unknown"


def _curve_slope_severity(config: Mapping[str, Any]) -> int:
    raw_values = [
        config.get("curveSlopeSeverity"),
        config.get("slopeSeverity"),
    ]
    existing_profile = config.get("strategyTuningProfile")
    if isinstance(existing_profile, Mapping):
        tuned_for = existing_profile.get("tunedFor")
        if isinstance(tuned_for, Mapping):
            direction = tuned_for.get("direction")
            if isinstance(direction, Mapping):
                raw_values.append(direction.get("curveSlopeSeverity"))
    for raw_value in raw_values:
        try:
            if raw_value is not None:
                return max(1, int(raw_value))
        except (TypeError, ValueError):
            continue
    return DEFAULT_CURVE_SLOPE_SEVERITY


def _field_matches(field: str, fragments: Sequence[str]) -> bool:
    lowered = field.lower()
    return any(fragment in lowered for fragment in fragments)


def _is_crypto(sec_type: str, symbol: str) -> bool:
    return sec_type == "CRYPTO" or symbol in CRYPTO_SYMBOLS or symbol.endswith("USD") and symbol[:-3] in CRYPTO_SYMBOLS
