from __future__ import annotations

from dataclasses import dataclass


DETECTOR_SPEC_SCHEMA = "regime_detector_specs.v1"
DETECTOR_SPEC_VERSION = "2026-06-01"
CORPUS_NOTE = "reports/regime_corpus_bfs_20260531/to_lookup_note.md"
ADVANCED_PARAMETERS_NOTE = (
    "reports/regime_corpus_bfs_20260531/advanced_regime_parameters.md"
)


@dataclass(frozen=True)
class DetectorPaper:
    title: str
    method_family: str
    corpus_key: str
    url: str

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "methodFamily": self.method_family,
            "corpusKey": self.corpus_key,
            "url": self.url,
        }


@dataclass(frozen=True)
class RegimeDetectorSpec:
    detector_id: str
    output_fields: tuple[str, ...]
    method_family: str
    required_inputs: tuple[str, ...]
    confidence_field: str
    implementation_status: str
    leakage_guardrail: str
    papers: tuple[DetectorPaper, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "detectorId": self.detector_id,
            "outputFields": list(self.output_fields),
            "methodFamily": self.method_family,
            "requiredInputs": list(self.required_inputs),
            "confidenceField": self.confidence_field,
            "implementationStatus": self.implementation_status,
            "leakageGuardrail": self.leakage_guardrail,
            "papers": [paper.to_dict() for paper in self.papers],
        }


DETECTOR_SPECS: tuple[RegimeDetectorSpec, ...] = (
    RegimeDetectorSpec(
        detector_id="direction_slope_persistence",
        output_fields=("directionSign", "slopeSeverity", "regimePersistence"),
        method_family="rolling_slope_with_markov_persistence_guard",
        required_inputs=("historical_bars.ohlc",),
        confidence_field="advanced.regimePersistence.score",
        implementation_status="heuristic_live_ready",
        leakage_guardrail=(
            "Use only bars whose timestamp is <= the detector evaluation time; "
            "switch active state only after hysteresis unless change-point confidence is high."
        ),
        papers=(
            DetectorPaper(
                title="Regime Changes and Financial Markets",
                method_family="Markov regime switching",
                corpus_key="nber:w17182",
                url="https://www.nber.org/papers/w17182",
            ),
            DetectorPaper(
                title=(
                    "Predicting Risk-adjusted Returns using an Asset Independent "
                    "Regime-switching Model"
                ),
                method_family="sticky HMM",
                corpus_key=(
                    "title:predicting risk adjusted returns using an asset "
                    "independent regime switching model"
                ),
                url="https://ideas.repec.org/p/arx/papers/2107.05535.html",
            ),
        ),
    ),
    RegimeDetectorSpec(
        detector_id="volatility_covariance_stress",
        output_fields=(
            "instrumentVolatilityRegime",
            "marketVolatilityRegime",
            "volatilitySpread",
            "covarianceStress",
        ),
        method_family="realized_volatility_and_covariance_regime",
        required_inputs=("historical_bars.ohlc", "market_proxy_bars.ohlc"),
        confidence_field="advanced.covarianceStress.observations",
        implementation_status="heuristic_live_ready",
        leakage_guardrail=(
            "Estimate volatility and covariance on the current completed window only; "
            "do not refit thresholds on future buckets."
        ),
        papers=(
            DetectorPaper(
                title="Market regime detection via realized covariances",
                method_family="realized covariance clustering",
                corpus_key="title:market regime detection via realized covariances",
                url="https://ideas.repec.org/a/eee/ecmode/v111y2022ics0264999322000785.html",
            ),
        ),
    ),
    RegimeDetectorSpec(
        detector_id="futures_alignment",
        output_fields=("indexFuturesDirection", "indexFuturesAlignment"),
        method_family="cross_asset_confirmation",
        required_inputs=("historical_bars.ohlc", "index_futures_bars.ohlc"),
        confidence_field="core.indexFuturesAlignment",
        implementation_status="heuristic_live_ready_when_futures_loaded",
        leakage_guardrail=(
            "Compare symbol and futures bars on aligned timestamps; stale futures "
            "data must produce not_loaded instead of inferred alignment."
        ),
        papers=(
            DetectorPaper(
                title="How do Regimes Affect Asset Allocation?",
                method_family="regime-switching allocation",
                corpus_key="nber:w10080",
                url="https://www.nber.org/papers/w10080",
            ),
        ),
    ),
    RegimeDetectorSpec(
        detector_id="momentum_distribution_change",
        output_fields=(
            "momentumHorizonRegime",
            "distributionClusterId",
            "changePointConfidence",
        ),
        method_family="momentum_horizons_plus_nonparametric_change_detection",
        required_inputs=("historical_bars.ohlc",),
        confidence_field="advanced.changePointConfidence.confidence",
        implementation_status="heuristic_now_model_backed_later",
        leakage_guardrail=(
            "Detect changes from prior-vs-recent completed windows; replay must "
            "advance chronologically and never cluster on the full sample."
        ),
        papers=(
            DetectorPaper(
                title="Time-Series Momentum",
                method_family="momentum horizon behavior",
                corpus_key="doi:10.1016/j.jfineco.2011.11.003",
                url="https://doi.org/10.1016/j.jfineco.2011.11.003",
            ),
            DetectorPaper(
                title="Clustering Market Regimes Using the Wasserstein Distance",
                method_family="Wasserstein distribution clustering",
                corpus_key="ssrn:3947905",
                url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905",
            ),
            DetectorPaper(
                title=(
                    "Non-parametric online market regime detection and regime "
                    "clustering for multidimensional and path-dependent data structures"
                ),
                method_family="online MMD/signature change detection",
                corpus_key="arxiv:2306.15835",
                url="https://arxiv.org/abs/2306.15835",
            ),
        ),
    ),
    RegimeDetectorSpec(
        detector_id="liquidity_orderflow",
        output_fields=("volumeRegime", "volumeDirection", "liquidityOrderFlowRegime"),
        method_family="volume_now_orderbook_later",
        required_inputs=("historical_bars.volume", "l2_orderbook_optional"),
        confidence_field="advanced.liquidityOrderFlowRegime.status",
        implementation_status="volume_live_ready_orderbook_pending",
        leakage_guardrail=(
            "Use current completed bar volume and live orderbook snapshots only; "
            "never derive order-flow state from post-trade hindsight."
        ),
        papers=(
            DetectorPaper(
                title=(
                    "Asymmetric Hidden Markov Modeling of Order Flow Imbalances "
                    "for Microstructure-Aware Market Regime Detection"
                ),
                method_family="order-flow HMM",
                corpus_key="ssrn:5315733",
                url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5315733",
            ),
        ),
    ),
    RegimeDetectorSpec(
        detector_id="options_surface",
        output_fields=("optionsSurfaceRegime",),
        method_family="options_probability_surface",
        required_inputs=("option_trades_or_surface",),
        confidence_field="advanced.optionsSurfaceRegime.status",
        implementation_status="not_loaded_until_options_seed_set_and_data_are_ready",
        leakage_guardrail=(
            "Use only option trades and surface snapshots observed before the "
            "decision timestamp; expired future information is prohibited."
        ),
        papers=(),
    ),
)


def detector_spec_manifest() -> dict[str, object]:
    return {
        "schema": DETECTOR_SPEC_SCHEMA,
        "version": DETECTOR_SPEC_VERSION,
        "corpusNote": CORPUS_NOTE,
        "advancedParametersNote": ADVANCED_PARAMETERS_NOTE,
        "specs": [spec.to_dict() for spec in DETECTOR_SPECS],
    }


def detector_specs_for_fields(fields: set[str]) -> list[dict[str, object]]:
    selected = []
    for spec in DETECTOR_SPECS:
        if fields.intersection(spec.output_fields):
            selected.append(spec.to_dict())
    return selected


def detector_ids_for_vector() -> list[str]:
    return [spec.detector_id for spec in DETECTOR_SPECS]
